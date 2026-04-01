from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, Callable, cast
from pathlib import Path
from typing import Any

from jcodemunch_mcp.tools.index_folder import index_folder

from .observability.langfuse import flush_langfuse, start_span, start_trace
from .observability.baselines import BaselineSnapshot, baseline_metrics_from_jc
from .observability.score_emitter import emit_score, emit_score_for_handle
from .pipeline import PipelineClassification, classify_results, default_pipeline
from .pipeline.types import StepResult
from .loop_machine import OptimizationStateMachine
from .loop_types import (
    AcceptedPolicyMeta,
    EvaluationResult,
    FailedRepairDetails,
    FeedbackPackage,
    IterationRecord,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
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


def _clean_policy_code(code: str) -> str:
    if "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if code.startswith("python"):
                code = code[len("python") :].lstrip()
    return code.strip()


def _hash_tests_dir(repo_root: Path) -> str:  # pyright: ignore[reportUnusedFunction]
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


def _step_succeeded(step: StepResult) -> bool:
    return step.exit_code == 0


def _first_failed_step(steps: Sequence[StepResult]) -> StepResult | None:
    for step in steps:
        if not _step_succeeded(step):
            return step
    return None


def _format_pipeline_failure_context(
    pipeline_results: list[StepResult],
    classified: PipelineClassification | None,
    writer_error: str | None,
    model_name: str | None,
    repair_attempt: int,
) -> str:
    failed_step = _first_failed_step(pipeline_results)
    pipeline_dump = [step.model_dump() for step in pipeline_results]
    classified_dump = classified.model_dump() if classified else None
    failed_dump = failed_step.model_dump() if failed_step else None
    return pack_failure_context(
        pipeline_dump,
        classified_dump,
        failed_dump,
        writer_error,
        model_name,
        repair_attempt=repair_attempt,
    )


class SupportsPipelineRun(Protocol):
    def run(self, repo_root: Path, observation: object | None = None) -> list[StepResult]:
        ...


class SupportsSpan(Protocol):
    def end(self, **kwargs: object) -> None:
        ...


def prepare_iteration_tasks(task: dict[str, object], parent_trace: object | None = None) -> PreparedTasks:
    """Resolve repo paths and index JC repo once for the run."""
    repo_path_val = task.get("repo")
    resolved_repo_path = repo_path_val if isinstance(repo_path_val, str) else None
    jc_repo_id: str | None = None
    index_obs: SupportsSpan | None = None
    if isinstance(repo_path_val, str):
        try:
            index_obs = start_span(parent_trace, "index_repo", metadata={"repo": repo_path_val})
            result = index_folder(repo_path_val)
            repo_id = str(result["repo"])
            jc_repo_id = repo_id
            if index_obs:
                index_obs.end(metadata={"repo": repo_id})
        except Exception as e:  # noqa: BLE001
            if index_obs:
                try:
                    index_obs.end(error=e)
                except Exception:
                    pass
    if resolved_repo_path is None:
        raise RuntimeError("Task missing repo")
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
        emit_score_for_handle(
            iteration_span,
            name="metrics.score",
            value=_as_float(metrics_map.get("score", 0.0)),
            data_type="NUMERIC",
        )
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

    ic_runner = cast(Callable[..., dict[str, object]], run_ic_iteration)
    ic_result = ic_runner(dict(ic_task), score_fn, parent_trace=iteration_span)
    jc_result: dict[str, object] = dict(baseline_snapshot.jc_result) if baseline_snapshot is not None else {}
    jc_metrics: dict[str, object] = dict(baseline_snapshot.jc_metrics) if baseline_snapshot is not None else {}
    if not jc_metrics:
        jc_metrics = {k: float(v) for k, v in baseline_metrics_from_jc(jc_result).items()}

    combined_result = {"iterative_context": ic_result, "jcodemunch": jc_result}
    metrics_map: dict[str, object] = dict(score(combined_result))
    if ic_result.get("node_count", 0) == 0:
        metrics_map["score"] = _as_float(metrics_map.get("score", 0.0)) - 1.0
        metrics_map["error"] = "no_tool_usage"

    total_ic, errors_ic = _tool_call_stats(ic_result)
    total_jc, errors_jc = _tool_call_stats(jc_result)
    total_calls = total_ic + total_jc
    total_errors = errors_ic + errors_jc
    metrics_map["tool_error_rate"] = (total_errors / total_calls) if total_calls else 0.0
    metrics_map["no_tool_usage"] = total_ic == 0
    comparison = build_comparison(metrics_map, jc_metrics)
    comparison_summary = format_comparison_summary(comparison)
    metrics_map.setdefault("iteration_regression", False)
    metrics_map["jc_baseline_metrics"] = dict(jc_metrics)
    metrics_map["comparison"] = comparison
    trace_id = getattr(iteration_span, "id", None)
    for name, value in metrics_map.items():
        if isinstance(value, (int, float, bool)):
            data_type = "BOOLEAN" if isinstance(value, bool) else "NUMERIC"
            score_id = f"{trace_id}-metrics.{name}" if trace_id else None
            emit_score_for_handle(
                iteration_span,
                name=f"metrics.{name}",
                value=float(value) if isinstance(value, bool) else value,
                data_type=data_type,
                score_id=score_id,
            )

    policy_code = _read_policy()
    return EvaluationResult(
        metrics=metrics_map,
        ic_result=ic_result,
        jc_result=jc_result,
        jc_metrics=jc_metrics,
        comparison_summary=comparison_summary,
        policy_code=policy_code,
    )


def build_iteration_feedback(
    evaluation: EvaluationResult,
    prev_score: float | None,
    prev_classified: PipelineClassification | None,
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
            diff = compute_diff(
                prev_score,
                _as_float(evaluation.metrics.get("score", 0.0)),
                prev_classified.model_dump(),
                prev_classified.model_dump(),
            )
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
    parent_trace: object | None = None,
    pipeline_feedback: PipelineClassification | None = None,
) -> tuple[str | None, str | None]:
    try:
        generator = cast(Callable[..., str], generate_policy)
        new_code = generator(
            feedback={
                **feedback,
                **({"pipeline_feedback": pipeline_feedback.model_dump()} if pipeline_feedback else {}),
            },
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


def run_policy_pipeline(pipeline: SupportsPipelineRun, repo_root: Path, repair_obs: object | None = None) -> tuple[list[StepResult], bool]:
    raw_results: list[object] = list(pipeline.run(repo_root, observation=repair_obs))
    step_results = [step if isinstance(step, StepResult) else StepResult.from_external(step) for step in raw_results]
    trace_id = getattr(repair_obs, "trace_id", None) if repair_obs is not None else None
    observation_id = getattr(repair_obs, "id", None) if repair_obs is not None else None
    dataset_run_id = getattr(repair_obs, "dataset_run_id", None) if repair_obs is not None else None
    for step in step_results:
        data_type = "NUMERIC"
        score_id = f"{observation_id or trace_id}-pipeline.{step.name}.exit_code" if (observation_id or trace_id) else None
        emit_score(
            name=f"pipeline.{step.name}.exit_code",
            value=step.exit_code,
            data_type=data_type,
            trace_id=trace_id,
            observation_id=observation_id,
            dataset_run_id=dataset_run_id,
            score_id=score_id,
        )
        pass_score_id = f"{observation_id or trace_id}-pipeline.{step.name}.passed" if (observation_id or trace_id) else None
        emit_score(
            name=f"pipeline.{step.name}.passed",
            value=int(step.success),
            data_type="BOOLEAN",
            trace_id=trace_id,
            observation_id=observation_id,
            dataset_run_id=dataset_run_id,
            score_id=pass_score_id,
        )
    pipeline_success = all(_step_succeeded(r) for r in step_results)
    return step_results, pipeline_success


def finalize_successful_policy(
    *,
    new_code: str,
    attempts_used: int,
    max_repair_attempts: int,
    repair_obs: object | None = None,
) -> tuple[str, AcceptedPolicyMeta]:
    final_policy = _read_policy()
    meta = AcceptedPolicyMeta(
        accepted_policy_source="post_pipeline_disk",
        accepted_policy_changed_by_pipeline=final_policy != new_code,
        repair_attempts=attempts_used - 1,
        max_policy_repairs=max_repair_attempts,
    )
    return final_policy, meta


def finalize_failed_repair(
    *,
    pipeline_results: list[StepResult],
    classified: PipelineClassification,
    writer_model: str | None,
    attempts_used: int,
    writer_error: str | None,
) -> FailedRepairDetails:
    failed_step = _first_failed_step(pipeline_results)
    failed_summary = None
    if failed_step:
        stderr = failed_step.stderr
        stdout = failed_step.stdout
        text = stderr.strip() if stderr.strip() else stdout.strip()
        failed_summary = text[:_MAX_FAILURE_SUMMARY_CHARS] if text else None
    failure_context_str = _format_pipeline_failure_context(
        pipeline_results, classified, writer_error, writer_model, attempts_used
    )
    return FailedRepairDetails(
        failure_context=failure_context_str,
        policy_code=_read_policy(),
        failed_step=failed_step.name if failed_step else None,
        failed_exit_code=failed_step.exit_code if failed_step else None,
        failed_summary=failed_summary,
        error="pipeline_failed",
    )


def _build_dependencies() -> LoopDependencies:
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
        start_span=start_span,
        find_repo_root=find_repo_root,
        default_pipeline=default_pipeline,
    )


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
    baseline_model = (
        BaselineSnapshot.model_validate(baseline_snapshot)
        if isinstance(baseline_snapshot, Mapping)
        else baseline_snapshot
    )
    context = LoopContext(
        task=dict(task),
        iterations=iterations,
        baseline_snapshot=baseline_model,
        run_trace=run_trace,
    )
    opt_model = OptimizationMachineModel(
        context=context,
        deps=_build_dependencies(),
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
    if not history:
        repo_root = find_repo_root()
        pipeline = default_pipeline()
        pipeline_results, pipeline_success = run_policy_pipeline(pipeline, repo_root)
        classified = classify_results(pipeline_results)
        failed_step = _first_failed_step(pipeline_results)
        record = IterationRecord(
            iteration=0,
            metrics={},
            pipeline_passed=pipeline_success,
            pipeline_feedback=classified,
            failed_step=failed_step.name if failed_step else "pipeline_failed",
            failed_exit_code=failed_step.exit_code if failed_step else None,
            failed_summary=(failed_step.stderr or failed_step.stdout) if failed_step else None,
            repair_attempts=_MAX_POLICY_REPAIRS,
            max_policy_repairs=_MAX_POLICY_REPAIRS,
            writer_error="writer_failed" if not pipeline_success else None,
        )
        history = [record]
    return history
