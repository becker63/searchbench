from __future__ import annotations

import difflib
import hashlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from pathlib import Path
from typing import Any

from jcodemunch_mcp.tools.index_folder import index_folder

from .observability.langfuse import flush_langfuse, record_score, start_span, start_trace
from .observability.baselines import BaselineSnapshot, baseline_metrics_from_jc
from .pipeline import classify_results, default_pipeline
from .pipeline.types import StepResult
from .loop_machine import OptimizationStateMachine, RepairStateMachine
from .loop_types import (
    FeedbackPackage,
    EvaluationResult,
    IterationRecord,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PipelineStepResult,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
    RepairOutcome,
)
from .policy_loader import load_policy
from .runner import run_ic_iteration
from .scorer import score
from .utils.comparison import build_comparison, format_comparison_summary
from .utils.diff import compute_diff, format_diff, interpret_diff
from .utils.env import get_writer_model
from .utils.repo_root import find_repo_root
from .utils.failure_context import pack_failure_context
from .utils.test_loader import format_tests_for_prompt, load_tests
from .utils.type_loader import format_scoring_context, load_scoring_examples, load_scoring_types
from .writer import generate_policy

if TYPE_CHECKING:
    from .observability.baselines import BaselineSnapshot

_POLICY_PATH = Path(__file__).with_name("policy.py")
_MAX_POLICY_REPAIRS = 3
_MAX_FAILURE_SUMMARY_CHARS = 400


def _as_float(value: Any) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _read_policy(path: Path = _POLICY_PATH) -> str:
    return path.read_text(encoding="utf-8")


def _write_policy(code: str, path: Path = _POLICY_PATH) -> None:
    if path.name != "policy.py":
        raise RuntimeError("Unauthorized write attempt")
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(code, encoding="utf-8")
    tmp_path.replace(path)


def print_policy_diff(old: str, new: str) -> None:
    if old == new:
        print("[POLICY] No change")
        return

    print("[POLICY] Updated")
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        lineterm="",
    )
    lines = list(diff)
    max_lines = 50
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print("... diff truncated")


def _clean_policy_code(code: str) -> str:
    if "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if code.startswith("python"):
                code = code[len("python") :].lstrip()
    return code.strip()


def _hash_tests_dir(repo_root: Path) -> str:
    tests_path = repo_root / "iterative-context" / "tests"
    hasher = hashlib.sha256()
    if not tests_path.exists():
        return hasher.hexdigest()
    for file in sorted(tests_path.rglob("*.py")):
        hasher.update(str(file.relative_to(repo_root)).encode())
        hasher.update(file.read_bytes())
    return hasher.hexdigest()


def _tool_call_stats(bucket: Mapping[str, object] | None) -> tuple[int, int]:
    if not isinstance(bucket, Mapping):
        return 0, 0
    observations = bucket.get("observations")
    if not isinstance(observations, list):
        return 0, 0
    errors = 0
    for obs in observations:
        if isinstance(obs, Mapping) and "error" in obs:
            errors += 1
    return len(observations), errors


def _step_succeeded(step: Mapping[str, object]) -> bool:
    if "exit_code" in step and isinstance(step["exit_code"], int):
        return step["exit_code"] == 0
    success_val = step.get("success")
    return bool(success_val)


def _first_failed_step(steps: Sequence[Mapping[str, object] | object]) -> Mapping[str, object] | None:
    def _as_mapping(step: object) -> Mapping[str, object] | None:
        if isinstance(step, Mapping):
            return step
        if step is None:
            return None
        data: dict[str, object] = {}
        for key in ("name", "success", "stdout", "stderr", "exit_code"):
            if hasattr(step, key):
                data[key] = getattr(step, key)
        return data if data else None

    for step in steps:
        step_mapping = _as_mapping(step)
        if not step_mapping:
            continue
        if not _step_succeeded(step_mapping):
            return step_mapping
    return None


def _compact_failure(step: Mapping[str, object]) -> str:
    stderr = step.get("stderr")
    stdout = step.get("stdout")
    text = ""
    if isinstance(stderr, str) and stderr.strip():
        text = stderr.strip()
    elif isinstance(stdout, str) and stdout.strip():
        text = stdout.strip()
    return text[:_MAX_FAILURE_SUMMARY_CHARS]


def _map_steps(steps: list[object]) -> list[Mapping[str, object]]:
    mapped: list[Mapping[str, object]] = []
    for step in steps:
        if isinstance(step, Mapping):
            mapped.append(step)
            continue
        data: dict[str, object] = {}
        for key in ("name", "success", "stdout", "stderr", "exit_code"):
            if hasattr(step, key):
                data[key] = getattr(step, key)
        if data:
            mapped.append(data)
    return mapped


def _format_pipeline_failure_context(
    pipeline_results: list[Mapping[str, object]],
    classified: Mapping[str, object] | None,
    writer_error: str | None,
    model_name: str | None,
    repair_attempt: int,
) -> str:
    failed_step = _first_failed_step(pipeline_results)
    return pack_failure_context(
        pipeline_results,
        classified,
        failed_step,
        writer_error,
        model_name,
        repair_attempt=repair_attempt,
    )


def prepare_iteration_tasks(task: Mapping[str, object], parent_trace=None) -> PreparedTasks:
    """Resolve repo paths and index JC repo once for the run."""
    repo_path = task.get("repo")
    resolved_repo_path = repo_path if isinstance(repo_path, str) else None
    jc_repo_id: str | None = None
    index_obs = None
    if isinstance(repo_path, str):
        try:
            index_obs = start_span(parent_trace, "index_repo", metadata={"repo": repo_path})
            result = index_folder(repo_path)
            repo_id = result["repo"]
            jc_repo_id = repo_id
            if index_obs and hasattr(index_obs, "end"):
                index_obs.end(metadata={"repo": repo_id})
        except Exception as e:  # noqa: BLE001
            if index_obs is not None and hasattr(index_obs, "end"):
                try:
                    index_obs.end(error=e)
                except Exception:
                    pass
    return PreparedTasks(base_task=dict(task), resolved_repo_path=resolved_repo_path, jc_repo_id=jc_repo_id)


def evaluate_policy_on_item(
    ic_task: Mapping[str, object],
    baseline_snapshot: BaselineSnapshot | None,
    iteration_span: object | None = None,
    iteration_index: int | None = None,
) -> EvaluationResult:
    """Load current policy, execute IC iteration, and compute metrics."""
    policy_load_obs = None
    try:
        policy_load_obs = start_span(iteration_span, "policy_load", metadata={"iteration": iteration_index})
        score_fn = load_policy()
        if policy_load_obs and hasattr(policy_load_obs, "end"):
            policy_load_obs.end(metadata={"status": "loaded"})
    except ImportError as e:
        if policy_load_obs is not None and hasattr(policy_load_obs, "end"):
            try:
                policy_load_obs.end(error=e)
            except Exception:
                pass
        metrics_map = {
            "score": -10.0,
            "error": f"policy_import_error: {str(e)}",
        }
        record_score(iteration_span, "metrics.score", _as_float(metrics_map.get("score", 0.0)))
        return EvaluationResult(
            metrics=metrics_map,
            ic_result={},
            jc_result={},
            jc_metrics={},
            comparison_summary=None,
            policy_code="",
            success=False,
            error=str(metrics_map.get("error")),
        )

    ic_result = run_ic_iteration(dict(ic_task), score_fn, parent_trace=iteration_span)
    jc_result = baseline_snapshot.jc_result if baseline_snapshot is not None else {}
    jc_metrics = baseline_snapshot.jc_metrics if baseline_snapshot is not None else {}
    if not isinstance(jc_metrics, Mapping):
        jc_metrics = baseline_metrics_from_jc(jc_result if isinstance(jc_result, Mapping) else {})

    combined_result = {"iterative_context": ic_result, "jcodemunch": jc_result}
    metrics_map: dict[str, object] = dict(score(combined_result))
    if ic_result.get("node_count", 0) == 0:
        metrics_map["score"] = _as_float(metrics_map.get("score", 0.0)) - 1.0
        metrics_map["error"] = "no_tool_usage"

    total_ic, errors_ic = _tool_call_stats(ic_result)
    total_jc, errors_jc = _tool_call_stats(jc_result if isinstance(jc_result, Mapping) else {})
    total_calls = total_ic + total_jc
    total_errors = errors_ic + errors_jc
    metrics_map["tool_error_rate"] = (total_errors / total_calls) if total_calls else 0.0
    metrics_map["no_tool_usage"] = total_ic == 0
    comparison = build_comparison(metrics_map, jc_metrics if isinstance(jc_metrics, Mapping) else {})
    comparison_summary = format_comparison_summary(comparison)
    metrics_map.setdefault("iteration_regression", False)
    metrics_map["jc_baseline_metrics"] = dict(jc_metrics) if isinstance(jc_metrics, Mapping) else {}
    metrics_map["comparison"] = comparison
    for name, value in metrics_map.items():
        if isinstance(value, (int, float, bool)):
            record_score(iteration_span, f"metrics.{name}", float(value))

    policy_code = _read_policy()
    return EvaluationResult(
        metrics=metrics_map,
        ic_result=ic_result,
        jc_result=jc_result if isinstance(jc_result, Mapping) else {},
        jc_metrics=jc_metrics if isinstance(jc_metrics, Mapping) else {},
        comparison_summary=comparison_summary,
        policy_code=policy_code,
    )


def build_iteration_feedback(
    evaluation: EvaluationResult,
    prev_score: float | None,
    prev_classified: dict[str, object] | None,
    repo_root: Path,
) -> FeedbackPackage:
    """Package evaluation outputs into writer-facing inputs."""
    tests = load_tests(repo_root)
    tests_str = format_tests_for_prompt(tests)
    types_str = load_scoring_types(repo_root)
    examples_str = load_scoring_examples(repo_root)
    scoring_ctx = format_scoring_context(types_str, examples_str)

    diff_str = ""
    diff_hint = ""
    if prev_score is not None and prev_classified is not None:
        try:
            diff = compute_diff(prev_score, _as_float(evaluation.metrics.get("score", 0.0)), prev_classified, prev_classified)
            diff_str = format_diff(diff)
            diff_hint = interpret_diff(diff)
        except Exception:
            diff_str = ""
            diff_hint = ""

    return FeedbackPackage(
        tests=tests_str,
        scoring_context=scoring_ctx,
        comparison_summary=evaluation.comparison_summary,
        feedback=dict(evaluation.metrics),
        feedback_str=str(evaluation.metrics.get("error", "")),
        guidance_hint="Fix errors before improving score",
        diff_str=diff_str,
        diff_hint=diff_hint,
    )


def attempt_policy_generation(
    *,
    feedback: dict[str, object],
    initial_policy: str,
    tests: str,
    scoring_context: str,
    feedback_str: str,
    guidance_hint: str,
    diff_str: str,
    diff_hint: str,
    comparison_summary: str | None,
    failure_context_str: str | None,
    repair_attempt: int,
    parent_trace=None,
    pipeline_feedback: Mapping[str, object] | None = None,
) -> tuple[str | None, str | None]:
    try:
        new_code = generate_policy(
            feedback={**feedback, **({"pipeline_feedback": pipeline_feedback} if pipeline_feedback else {})},
            current_policy=initial_policy,
            tests=tests,
            scoring_context=scoring_context,
            feedback_str=str(feedback_str),
            guidance_hint=guidance_hint,
            diff_str=diff_str,
            diff_hint=diff_hint,
            comparison_summary=comparison_summary,
            parent_trace=parent_trace,
            failure_context=failure_context_str,
            repair_attempt=repair_attempt,
        )
        new_code = _clean_policy_code(new_code)
        return new_code, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def run_policy_pipeline(pipeline, repo_root: Path, repair_obs: object | None = None) -> tuple[list[PipelineStepResult], list[StepResult], bool]:
    step_results = pipeline.run(repo_root, observation=repair_obs)
    pipeline_results = [PipelineStepResult.from_mapping(r) for r in _map_steps(list(step_results))]
    normalized_results = pipeline_results or [PipelineStepResult.from_mapping(r) for r in _map_steps(list(step_results))]
    pipeline_success = all(_step_succeeded(r.to_mapping()) for r in normalized_results)
    return pipeline_results, list(step_results), pipeline_success


def finalize_successful_policy(
    *,
    new_code: str,
    attempts_used: int,
    max_repair_attempts: int,
    repair_obs=None,
) -> tuple[str, dict[str, object]]:
    final_policy = _read_policy()
    metadata: dict[str, object] = {
        "repair_attempts": attempts_used - 1,
        "max_policy_repairs": max_repair_attempts,
        "accepted_policy_source": "post_pipeline_disk",
        "accepted_policy_changed_by_pipeline": final_policy != new_code,
    }
    return final_policy, metadata


def finalize_failed_repair(
    *,
    metadata: dict[str, object],
    pipeline_results: list[Mapping[str, object]],
    classified: Mapping[str, object],
    writer_model: str | None,
    attempts_used: int,
) -> tuple[str, str]:
    failed_step = _first_failed_step(pipeline_results)
    if failed_step:
        metadata["failed_step"] = failed_step.get("name")
        metadata["failed_exit_code"] = failed_step.get("exit_code")
        metadata["failed_summary"] = _compact_failure(failed_step)
    metadata["pipeline_feedback"] = classified
    metadata["error"] = "pipeline_failed"
    writer_err_val = metadata.get("writer_error")
    writer_err_str = str(writer_err_val) if writer_err_val is not None else None
    failure_context_str = _format_pipeline_failure_context(pipeline_results, classified, writer_err_str, writer_model, attempts_used)
    return failure_context_str, _read_policy()


def _build_dependencies(repair_override=None) -> LoopDependencies:
    return LoopDependencies(
        prepare_iteration_tasks=prepare_iteration_tasks,
        evaluate_policy_on_item=evaluate_policy_on_item,
        build_iteration_feedback=build_iteration_feedback,
        attempt_policy_generation=attempt_policy_generation,
        run_policy_pipeline=run_policy_pipeline,
        finalize_successful_policy=finalize_successful_policy,
        finalize_failed_repair=finalize_failed_repair,
        classify_results=lambda results: classify_results(list(results)),
        format_pipeline_failure_context=_format_pipeline_failure_context,
        read_policy=_read_policy,
        write_policy=_write_policy,
        get_writer_model=get_writer_model,
        record_score=record_score,
        start_span=start_span,
        find_repo_root=find_repo_root,
        default_pipeline=default_pipeline,
        repair_via_synthesize=repair_override,
    )


def synthesize_valid_policy(
    *,
    repo_root: Path,
    initial_policy: str,
    feedback: dict[str, object],
    tests: str,
    scoring_context: str,
    feedback_str: str,
    guidance_hint: str,
    diff_str: str,
    diff_hint: str,
    comparison_summary: str | None,
    parent_trace: object | None = None,
    max_repair_attempts: int = _MAX_POLICY_REPAIRS,
) -> tuple[str, list[dict[str, object]], bool, int, dict[str, object]]:
    """Compatibility wrapper that reuses the repair state machine."""
    evaluation = EvaluationResult(
        metrics=dict(feedback),
        ic_result={},
        jc_result={},
        jc_metrics={},
        comparison_summary=comparison_summary,
        policy_code=initial_policy,
    )
    feedback_pkg = FeedbackPackage(
        tests=tests,
        scoring_context=scoring_context,
        comparison_summary=comparison_summary,
        feedback=dict(feedback),
        feedback_str=feedback_str,
        guidance_hint=guidance_hint,
        diff_str=diff_str,
        diff_hint=diff_hint,
    )
    repair_ctx = RepairContext(
        repo_root=repo_root,
        evaluation=evaluation,
        feedback=feedback_pkg,
        max_repair_attempts=max_repair_attempts,
        parent_trace=parent_trace,
        writer_model=get_writer_model(),
        pipeline=default_pipeline(),
    )
    repair_model = RepairMachineModel(
        context=repair_ctx,
        deps=_build_dependencies(),
    )
    repair_machine = RepairStateMachine(repair_model)
    outcome = repair_machine.run()
    pipeline_results = [r.to_mapping() for r in outcome.pipeline_results]
    return outcome.policy, pipeline_results, outcome.success, outcome.attempts_used, outcome.metadata


_ORIGINAL_SYNTHESIZE = synthesize_valid_policy


def run_loop(
    task: Mapping[str, object],
    iterations: int = 5,
    parent_trace: object | None = None,
    baseline_snapshot: BaselineSnapshot | None = None,
) -> list[IterationRecord]:
    """
    Closed-loop optimizer: execute, score, rewrite policy, repeat.
    """
    root_meta = {"task": dict(task), "iterations": iterations}
    run_trace = start_span(parent_trace, "run_loop", metadata=root_meta, input=task) if parent_trace else start_trace("run_loop", metadata=root_meta, input=task)
    repair_override = None
    if synthesize_valid_policy is not _ORIGINAL_SYNTHESIZE:
        def _compat_repair(evaluation: EvaluationResult, feedback: FeedbackPackage, repo_root: Path, parent_trace: object | None) -> RepairOutcome:
            policy, pipeline_results, pipeline_success, attempts, metadata = synthesize_valid_policy(  # type: ignore[call-arg]
                repo_root=repo_root,
                initial_policy=evaluation.policy_code,
                feedback=feedback.feedback,
                tests=feedback.tests,
                scoring_context=feedback.scoring_context,
                feedback_str=feedback.feedback_str,
                guidance_hint=feedback.guidance_hint,
                diff_str=feedback.diff_str,
                diff_hint=feedback.diff_hint,
                comparison_summary=feedback.comparison_summary,
                parent_trace=parent_trace,
                max_repair_attempts=_MAX_POLICY_REPAIRS,
            )
            return RepairOutcome(
                policy=policy,
                pipeline_results=[PipelineStepResult.from_mapping(r) for r in pipeline_results],
                success=pipeline_success,
                attempts_used=attempts,
                metadata=metadata,
            )

        repair_override = _compat_repair
    baseline_model = (
        BaselineSnapshot.model_validate(baseline_snapshot)
        if isinstance(baseline_snapshot, Mapping)
        else baseline_snapshot
    )
    context = LoopContext(
        task=task,
        iterations=iterations,
        baseline_snapshot=baseline_model,
        run_trace=run_trace,
    )
    opt_model = OptimizationMachineModel(
        context=context,
        deps=_build_dependencies(repair_override),
        max_policy_repairs=_MAX_POLICY_REPAIRS,
    )
    machine = OptimizationStateMachine(opt_model)
    history: list[IterationRecord] = []
    try:
        history = machine.run()
    except KeyboardInterrupt:
        if run_trace and hasattr(run_trace, "end"):
            try:
                run_trace.end(error="interrupted")
            except Exception:
                pass
    finally:
        if run_trace and hasattr(run_trace, "end"):
            try:
                run_trace.end(metadata={"iterations_completed": len(history)})
            except Exception:
                pass
        flush_langfuse()
    return history
