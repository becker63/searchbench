from __future__ import annotations

"""
Single-run optimizer coordinator: build loop context/deps and invoke the optimization machine.
Does not parse CLI or dataset concerns; expects normalized inputs and resolved session_id.
"""

import hashlib
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Protocol, cast

from jcodemunch_mcp.tools.index_folder import index_folder

from harness.localization.agent_runtime import run_ic_iteration
from harness.telemetry.langfuse import (
    flush_langfuse,
    propagate_context,
    start_observation,
)
from harness.telemetry.score_emitter import emit_score
from harness.pipeline import PipelineClassification, classify_results, default_pipeline
from harness.pipeline.types import StepResult
from harness.policy.load import load_policy
from harness.utils.diff import compute_diff, format_diff, interpret_diff
from harness.utils.env import get_writer_model
from harness.utils.repo_root import find_repo_root
from harness.utils.test_loader import format_tests_for_prompt, load_tests
from harness.utils.type_loader import ScorerContext, build_scorer_context, format_scoring_context, load_graph_models, load_scoring_examples, load_scoring_types
from harness.agents.writer import generate_policy
from harness.localization.models import LCAContext, LCATask, LCATaskIdentity, LCAGold
from harness.orchestration.types import SpanHandle

if TYPE_CHECKING:
    from harness.localization.evaluate import LocalizationEvaluationRequest as LocalizationEvaluationRequestType
    from harness.telemetry.baselines import BaselineSnapshot
    from iterative_context.graph_models import Graph, GraphNode
else:
    LocalizationEvaluationRequestType = object

from .machine import OptimizationStateMachine
from .types import (
    AcceptedPolicyMeta,
    EvaluationMetrics,
    EvaluationResult,
    FailedRepairDetails,
    FeedbackEntries,
    FeedbackEntry,
    FeedbackPackage,
    ICResult,
    ICTaskPayload,
    IterationRecord,
    JCResult,
    JCTaskPayload,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
    RunMetadata,
    TaskPayload,
    format_failure_context_payload,
)

_POLICY_PATH = Path(__file__).resolve().parent.parent / "policy" / "current.py"
_MAX_POLICY_REPAIRS = 3
_MAX_FAILURE_SUMMARY_CHARS = 400


# Re-exported helpers so callers can monkeypatch harness.loop.* directly.
_REEXPORTED = (
    index_folder,
    load_policy,
    load_graph_models,
    run_ic_iteration,
    compute_diff,
    format_diff,
    interpret_diff,
    get_writer_model,
    find_repo_root,
    format_tests_for_prompt,
    load_tests,
    build_scorer_context,
    format_scoring_context,
    load_scoring_examples,
    load_scoring_types,
    generate_policy,
    classify_results,
    default_pipeline,
    flush_langfuse,
    emit_score,
)


def evaluate_localization_batch(req: "LocalizationEvaluationRequestType"):
    """
    Thin wrapper to keep monkeypatching stable while avoiding import cycles.
    """
    from harness.localization.evaluate import evaluate_localization_batch as _evaluate_localization_batch

    return _evaluate_localization_batch(req)


def LocalizationEvaluationRequest(
    *args: object, **kwargs: object
) -> "LocalizationEvaluationRequestType":
    """
    Lazy proxy for the evaluation request model so tests can monkeypatch the symbol on this module.
    """
    from harness.localization.evaluate import LocalizationEvaluationRequest as _LocalizationEvaluationRequest

    return _LocalizationEvaluationRequest(*args, **kwargs)


def score(node: "GraphNode", graph: "Graph", step: int) -> float:
    """
    Delegate to policy.score while keeping a stable harness.loop.* attribute for tests/monkeypatching.
    """
    from harness.policy import score as _policy_score

    return _policy_score(node, graph, step)


def _pkg_attr(name: str) -> Any:
    mod = import_module("harness.orchestration.runtime")
    return getattr(mod, name)


def _as_float(value: Any) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _read_policy(path: Path = _POLICY_PATH) -> str:
    return path.read_text(encoding="utf-8")


def _write_policy(code: str, path: Path = _POLICY_PATH) -> None:  # pyright: ignore[reportUnusedFunction]
    if path.name not in ("policy.py", "current.py"):
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
    return format_failure_context_payload(
        pipeline_results,
        classified,
        writer_error,
        model_name,
        repair_attempt,
    )


class SupportsPipelineRun(Protocol):
    def run(
        self,
        repo_root: Path,
        observation: SpanHandle | None = None,
        allow_no_parent: bool = False,
    ) -> list[StepResult]: ...


def prepare_iteration_tasks(
    task: TaskPayload, parent_trace: SpanHandle | None = None
) -> PreparedTasks:
    """Resolve repo paths and index JC repo once for the run."""
    repo_path_val = task.repo
    resolved_repo_path = repo_path_val
    jc_repo_id: str | None = None
    start_obs_func: Callable[..., ContextManager[SpanHandle]] = _pkg_attr("start_observation")
    index_folder_fn: Callable[[str], Mapping[str, object]] = _pkg_attr("index_folder")
    if isinstance(repo_path_val, str):
        try:
            with start_obs_func(
                name="index_repo",
                parent=parent_trace,
                metadata={"repo": repo_path_val},
            ) as index_obs:
                result = index_folder_fn(repo_path_val)
                repo_id = str(result.get("repo", ""))
                jc_repo_id = repo_id
                if hasattr(index_obs, "update"):
                    try:
                        index_obs.update(metadata={"repo": repo_id})  # type: ignore[call-arg]
                    except Exception:
                        pass
        except Exception:
            # Propagate failure; lifecycle handled by context manager.
            raise
    if resolved_repo_path is None:
        raise RuntimeError("Task missing repo")
    ic_task = ICTaskPayload(
        identity=task.identity,
        context=task.context,
        repo=resolved_repo_path,
        changed_files=task.changed_files,
    )
    jc_task = JCTaskPayload(
        identity=task.identity,
        context=task.context,
        repo=jc_repo_id or resolved_repo_path,
        changed_files=task.changed_files,
    )
    return PreparedTasks(
        base_task=task,
        resolved_repo_path=resolved_repo_path,
        jc_repo_id=jc_repo_id,
        ic_task=ic_task,
        jc_task=jc_task,
    )


def evaluate_policy_on_item(
    ic_task: ICTaskPayload,
    baseline_snapshot: "BaselineSnapshot | None",
    iteration_span: object | None = None,
    iteration_index: int | None = None,
) -> EvaluationResult:
    """Execute localization evaluation via the shared LCA evaluation backend."""
    lca_task = LCATask(
        identity=LCATaskIdentity(
            dataset_name=ic_task.identity.dataset_name,
            dataset_config=ic_task.identity.dataset_config,
            dataset_split=ic_task.identity.dataset_split,
            repo_owner=ic_task.identity.repo_owner,
            repo_name=ic_task.identity.repo_name,
            base_sha=ic_task.identity.base_sha,
            issue_url=ic_task.identity.issue_url,
            pull_url=ic_task.identity.pull_url,
        ),
        context=LCAContext.model_validate(ic_task.context.model_dump()),
        gold=LCAGold(changed_files=list(ic_task.changed_files)),
        repo=ic_task.repo,
    )
    eval_req = LocalizationEvaluationRequest(
        tasks=[lca_task],
        dataset_source=None,
        materializer=None,
        parent_trace=iteration_span,
    )
    eval_result = evaluate_localization_batch(eval_req)
    if eval_result.failure:
        failure_metrics: dict[str, float | int | bool | None] = {
            "score": -10.0,
            "iteration_regression": False,
        }
        return EvaluationResult(
            metrics=EvaluationMetrics.model_validate(failure_metrics),
            ic_result=ICResult(entries=[]),
            jc_result=JCResult(entries=[]),
            jc_metrics=EvaluationMetrics(),
            comparison_summary=None,
            policy_code="",
            success=False,
            error=f"{getattr(eval_result.failure.category, 'value', eval_result.failure.category)}: {eval_result.failure.message}",
        )
    if not eval_result.items:
        empty_metrics: dict[str, float | int | bool | None] = {
            "score": -10.0,
            "iteration_regression": False,
        }
        return EvaluationResult(
            metrics=EvaluationMetrics.model_validate(empty_metrics),
            ic_result=ICResult(entries=[]),
            jc_result=JCResult(entries=[]),
            jc_metrics=EvaluationMetrics(),
            comparison_summary=None,
            policy_code="",
            success=False,
            error="evaluation_failed: no_results",
    )
    task_result = eval_result.items[0]
    metrics_obj = task_result.metrics
    machine_score_val = eval_result.machine_score if eval_result.machine_score is not None else metrics_obj.score
    additional_metrics = [
        {"name": "precision", "value": metrics_obj.precision},
        {"name": "recall", "value": metrics_obj.recall},
        {"name": "hit", "value": metrics_obj.hit},
        {"name": "f1", "value": metrics_obj.f1},
    ]
    eval_metrics = EvaluationMetrics.model_validate(
        {"score": metrics_obj.score, "iteration_regression": False, "additional_metrics": additional_metrics}
    )
    ic_result_map = {"predicted_files": task_result.prediction}
    return EvaluationResult(
        metrics=eval_metrics,
        ic_result=ICResult.model_validate(ic_result_map),
        jc_result=JCResult(entries=[]),
        jc_metrics=EvaluationMetrics(),
        comparison_summary=None,
        policy_code=_read_policy(),
        control_score=machine_score_val,
    )


def build_iteration_feedback(
    evaluation: EvaluationResult,
    prev_score: float | None,
    prev_classified: PipelineClassification | None,
    repo_root: Path,
) -> FeedbackPackage:
    """Package evaluation outputs into writer-facing inputs."""
    load_tests_fn = cast(Callable[[Path], object], _pkg_attr("load_tests"))
    format_tests_for_prompt_fn = cast(
        Callable[[object], str], _pkg_attr("format_tests_for_prompt")
    )
    build_scorer_context_fn = cast(Callable[[Path], ScorerContext], _pkg_attr("build_scorer_context"))
    format_scoring_context_fn = cast(Callable[..., str], _pkg_attr("format_scoring_context"))
    compute_diff_fn = cast(Callable[..., object], _pkg_attr("compute_diff"))
    format_diff_fn = cast(Callable[[object], str], _pkg_attr("format_diff"))
    interpret_diff_fn = cast(Callable[[object], str], _pkg_attr("interpret_diff"))

    tests = load_tests_fn(repo_root)
    tests_str = format_tests_for_prompt_fn(tests)
    scorer_ctx = build_scorer_context_fn(repo_root)
    scoring_ctx = format_scoring_context_fn(scorer_ctx)

    diff_str = ""
    diff_hint = ""
    if prev_score is not None and prev_classified is not None:
        try:
            diff = compute_diff_fn(
                prev_score,
                _as_float(evaluation.metrics.get("score", 0.0)),
                prev_classified.model_dump(),
                prev_classified.model_dump(),
            )
            diff_str = format_diff_fn(diff)
            diff_hint = interpret_diff_fn(diff)
        except Exception:
            diff_str = ""
            diff_hint = ""

    feedback_entries: list[FeedbackEntry] = []
    for key, value in evaluation.metrics.items():
        normalized_val: str | float | int | bool | None
        if isinstance(value, (str, float, int, bool)) or value is None:
            normalized_val = value
        else:
            normalized_val = str(value)
        feedback_entries.append(FeedbackEntry(name=key, value=normalized_val))

    return FeedbackPackage(
        tests=tests_str,
        scoring_context=scoring_ctx,
        scoring_context_details=scorer_ctx,
        comparison_summary=evaluation.comparison_summary,
        feedback=FeedbackEntries.model_validate({"entries": feedback_entries}),
        feedback_str=str(evaluation.metrics.get("error", "")),
        guidance_hint="Fix errors before improving score",
        diff_str=diff_str,
        diff_hint=diff_hint,
    )


def attempt_policy_generation(
    *,
    feedback: FeedbackEntries,
    initial_policy: str,
    tests: str,
    scoring_context: str,
    scoring_context_details: ScorerContext,
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
        generator = cast(Callable[..., str], _pkg_attr("generate_policy"))
        new_code = generator(
            feedback={
                **feedback.as_map(),
                **(
                    {"pipeline_feedback": pipeline_feedback.model_dump()}
                    if pipeline_feedback
                    else {}
                ),
            },
            current_policy=initial_policy,
            tests=tests,
            scoring_context=scoring_context,
            scoring_context_details=scoring_context_details,
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


def run_policy_pipeline(
    pipeline: SupportsPipelineRun,
    repo_root: Path,
    repair_obs: SpanHandle | None = None,
    allow_no_parent: bool = False,
) -> tuple[list[StepResult], bool]:
    try:
        raw_results: list[object] = list(
            pipeline.run(
                repo_root, observation=repair_obs, allow_no_parent=allow_no_parent
            )
        )
    except TypeError:
        raw_results = list(pipeline.run(repo_root, observation=repair_obs))
    step_results = [
        step if isinstance(step, StepResult) else StepResult.from_external(step)
        for step in raw_results
    ]
    trace_id = getattr(repair_obs, "trace_id", None) if repair_obs is not None else None
    observation_id = getattr(repair_obs, "id", None) if repair_obs is not None else None
    session_id_val = (
        getattr(repair_obs, "session_id", None) if repair_obs is not None else None
    )
    emit_score_fn = cast(Callable[..., None], _pkg_attr("emit_score"))
    if not any([trace_id, observation_id, session_id_val]):
        pipeline_success = all(_step_succeeded(r) for r in step_results)
        return step_results, pipeline_success
    for step in step_results:
        data_type = "NUMERIC"
        score_id = (
            f"{observation_id or trace_id}-pipeline.{step.name}.exit_code"
            if (observation_id or trace_id)
            else None
        )
        emit_score_fn(
            name=f"pipeline.{step.name}.exit_code",
            value=step.exit_code,
            data_type=data_type,
            trace_id=trace_id,
            observation_id=observation_id,
            session_id=session_id_val,
            score_id=score_id,
        )
        pass_score_id = (
            f"{observation_id or trace_id}-pipeline.{step.name}.passed"
            if (observation_id or trace_id)
            else None
        )
        emit_score_fn(
            name=f"pipeline.{step.name}.passed",
            value=int(step.success),
            data_type="BOOLEAN",
            trace_id=trace_id,
            observation_id=observation_id,
            session_id=session_id_val,
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
    classify_results_fn: Callable[[list[StepResult]], PipelineClassification] = _pkg_attr("classify_results")
    format_pipeline_failure_context_fn = _format_pipeline_failure_context
    read_policy_fn = _pkg_attr("_read_policy")
    write_policy_fn = _pkg_attr("_write_policy")
    get_writer_model_fn: Callable[[], str | None] = _pkg_attr("get_writer_model")
    start_obs_fn: Callable[..., ContextManager[SpanHandle]] = _pkg_attr("start_observation")
    find_repo_root_fn: Callable[[], Path] = _pkg_attr("find_repo_root")
    default_pipeline_fn: Callable[[], SupportsPipelineRun] = _pkg_attr("default_pipeline")
    return LoopDependencies(
        prepare_iteration_tasks=prepare_iteration_tasks,
        evaluate_policy_on_item=evaluate_policy_on_item,
        build_iteration_feedback=build_iteration_feedback,
        attempt_policy_generation=attempt_policy_generation,
        run_policy_pipeline=run_policy_pipeline,
        finalize_successful_policy=finalize_successful_policy,
        finalize_failed_repair=finalize_failed_repair,
        classify_results=lambda results: classify_results_fn(list(results)),
        format_pipeline_failure_context=format_pipeline_failure_context_fn,
        read_policy=cast(Callable[[], str], read_policy_fn),
        write_policy=cast(Callable[[str], None], write_policy_fn),
        get_writer_model=get_writer_model_fn,
        start_observation=start_obs_fn,
        find_repo_root=find_repo_root_fn,
        default_pipeline=default_pipeline_fn,
    )


def run_loop(
    task: Mapping[str, object] | TaskPayload,
    iterations: int = 5,
    parent_trace: SpanHandle | None = None,
    baseline_snapshot: BaselineSnapshot | None = None,
    session_id: str | None = None,
) -> list[IterationRecord]:
    """
    Closed-loop optimizer: execute, score, rewrite policy, repeat.
    """
    base_task = TaskPayload.model_validate(task)
    root_meta = RunMetadata(
        task=base_task, iterations=iterations, session_id=session_id
    )
    with start_observation(
        name="run_loop",
        parent=parent_trace,
        input=base_task.model_dump(),
        metadata=root_meta.model_dump(),
    ) as run_span:
        with propagate_context(
            trace_name="run_loop",
            session_id=session_id,
            metadata={"iterations": iterations, "session_id": session_id},
        ):
            baseline_model = (
                BaselineSnapshot.model_validate(baseline_snapshot)
                if isinstance(baseline_snapshot, Mapping)
                else baseline_snapshot
            )
            context = LoopContext(
                task=base_task,
                iterations=iterations,
                session_id=session_id,
                parent_trace=parent_trace,
                run_metadata=root_meta,
                baseline_snapshot=baseline_model,
                run_trace=run_span,
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
            finally:
                if hasattr(run_span, "update"):
                    try:
                        run_span.update(metadata={"iterations_completed": len(history)})  # type: ignore[call-arg]
                    except Exception:
                        pass
                flush_langfuse_fn = cast(Callable[[], None], _pkg_attr("flush_langfuse"))
                flush_langfuse_fn()
    if not history:
        repo_root = cast(Callable[[], Path], _pkg_attr("find_repo_root"))()
        pipeline = cast(
            Callable[[], SupportsPipelineRun], _pkg_attr("default_pipeline")
        )()
        fallback_parent = opt_model.context.run_trace or parent_trace
        pipeline_results, pipeline_success = run_policy_pipeline(
            pipeline,
            repo_root,
            repair_obs=fallback_parent,
            allow_no_parent=fallback_parent is None,
        )
        classify_results_fn = cast(
            Callable[[list[StepResult]], PipelineClassification],
            _pkg_attr("classify_results"),
        )
        classified = classify_results_fn(pipeline_results)
        failed_step = _first_failed_step(pipeline_results)
        metrics_map: dict[str, float | int | bool | None] = {}
        record = IterationRecord(
            iteration=0,
            metrics=EvaluationMetrics.model_validate(metrics_map),
            pipeline_passed=pipeline_success,
            pipeline_feedback=classified,
            failed_step=failed_step.name if failed_step else "pipeline_failed",
            failed_exit_code=failed_step.exit_code if failed_step else None,
            failed_summary=(failed_step.stderr or failed_step.stdout)
            if failed_step
            else None,
            repair_attempts=_MAX_POLICY_REPAIRS,
            max_policy_repairs=_MAX_POLICY_REPAIRS,
            writer_error="writer_failed" if not pipeline_success else None,
        )
        history = [record]
    return history
