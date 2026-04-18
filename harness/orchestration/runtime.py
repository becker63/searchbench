"""
Single-run optimizer coordinator.

This module owns the public loop entrypoint, dependency assembly, state-machine
startup, fallback record construction, and compatibility patch points.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from jcodemunch_mcp.tools.index_folder import index_folder

from harness.agents.localizer import run_ic_iteration
from harness.agents.writer import generate_policy
from harness.localization.materialization.materialize import RepoMaterializer
from harness.localization.models import LCATask
from harness.localization.runtime.evaluate import (
    LocalizationEvaluationResult,
    LocalizationRunner,
)
from harness.pipeline import PipelineClassification, classify_results, default_pipeline
from harness.pipeline.types import StepResult
from harness.policy.load import load_frontier_policy
from harness.telemetry.tracing import flush_langfuse, propagate_context, start_observation
from harness.telemetry.tracing.score_emitter import emit_score
from harness.utils.diff import compute_diff, format_diff, interpret_diff
from harness.utils.env import get_writer_model
from harness.utils.repo_root import find_repo_root
from harness.utils.test_loader import format_tests_for_prompt, load_tests
from harness.utils.type_loader import (
    build_frontier_context,
    format_frontier_context,
    load_frontier_examples,
    load_frontier_types,
    load_graph_models,
)

from . import dependencies as dependency_wiring
from .evaluation import evaluate_policy_on_item
from .feedback import build_iteration_feedback
from .machine import OptimizationStateMachine
from .types import (
    AcceptedPolicyMeta,
    EvaluationMetrics,
    FailedRepairDetails,
    IterationRecord,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
    RunMetadata,
    SpanHandle,
    SupportsPipelineRun,
)
from .writer import attempt_policy_generation, clean_policy_code

if TYPE_CHECKING:
    from harness.telemetry.hosted.baselines import BaselineSnapshot
    from iterative_context.graph_models import Graph, GraphNode


_POLICY_PATH = Path(__file__).resolve().parent.parent / "policy" / "current.py"
_MAX_POLICY_REPAIRS = 3


# Compatibility exports used by tests and legacy monkeypatch call sites.
_COMPATIBILITY_EXPORTS = (
    index_folder,
    load_frontier_policy,
    load_graph_models,
    load_frontier_examples,
    load_frontier_types,
    run_ic_iteration,
    compute_diff,
    format_diff,
    interpret_diff,
    get_writer_model,
    find_repo_root,
    format_tests_for_prompt,
    load_tests,
    build_frontier_context,
    format_frontier_context,
    generate_policy,
    classify_results,
    default_pipeline,
    flush_langfuse,
    emit_score,
)


def evaluate_localization_batch(
    tasks: Sequence[LCATask],
    *,
    dataset_source: str | None = None,
    materializer: RepoMaterializer | None = None,
    parent_trace: object | None = None,
    runner: LocalizationRunner | None = None,
) -> LocalizationEvaluationResult:
    """
    Stable wrapper for tests and callers that patch runtime-level evaluation.
    """
    from harness.localization.runtime.evaluate import evaluate_localization_batch as _evaluate_localization_batch

    return _evaluate_localization_batch(
        tasks,
        dataset_source=dataset_source,
        materializer=materializer,
        parent_trace=parent_trace,
        runner=runner,
    )


def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """
    Stable wrapper around the active policy function.
    """
    from harness.policy import frontier_priority as _frontier_priority

    return _frontier_priority(node, graph, step)


def _read_policy(path: Path = _POLICY_PATH) -> str:
    return path.read_text(encoding="utf-8")


def _write_policy(code: str, path: Path = _POLICY_PATH) -> None:
    if path.name not in ("policy.py", "current.py"):
        raise RuntimeError("Unauthorized write attempt")
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(code, encoding="utf-8")
    tmp_path.replace(path)


def _clean_policy_code(code: str) -> str:
    return clean_policy_code(code)


def _hash_tests_dir(repo_root: Path) -> str:
    tests_path = repo_root / "iterative-context" / "tests"
    hasher = hashlib.sha256()
    if not tests_path.exists():
        return hasher.hexdigest()
    for file in sorted(tests_path.rglob("*.py")):
        hasher.update(str(file.relative_to(repo_root)).encode())
        hasher.update(file.read_bytes())
    return hasher.hexdigest()


def prepare_iteration_tasks(
    task: LCATask,
    parent_trace: SpanHandle | None = None,
) -> PreparedTasks:
    return dependency_wiring.prepare_iteration_tasks(
        task,
        parent_trace,
        start_observation=start_observation,
        index_folder=index_folder,
    )


def run_policy_pipeline(
    pipeline: SupportsPipelineRun,
    repo_root: Path,
    repair_obs: SpanHandle | None = None,
    allow_no_parent: bool = False,
) -> tuple[list[StepResult], bool]:
    return dependency_wiring.run_policy_pipeline(
        pipeline,
        repo_root,
        repair_obs=repair_obs,
        allow_no_parent=allow_no_parent,
        emit_score=emit_score,
    )


def finalize_successful_policy(
    *,
    new_code: str,
    attempts_used: int,
    max_repair_attempts: int,
    repair_obs: object | None = None,
) -> tuple[str, AcceptedPolicyMeta]:
    return dependency_wiring.finalize_successful_policy(
        new_code=new_code,
        attempts_used=attempts_used,
        max_repair_attempts=max_repair_attempts,
        repair_obs=repair_obs,
        read_policy=_read_policy,
    )


def finalize_failed_repair(
    *,
    pipeline_results: list[StepResult],
    classified: PipelineClassification,
    writer_model: str | None,
    attempts_used: int,
    writer_error: str | None,
) -> FailedRepairDetails:
    return dependency_wiring.finalize_failed_repair(
        pipeline_results=pipeline_results,
        classified=classified,
        writer_model=writer_model,
        attempts_used=attempts_used,
        writer_error=writer_error,
        read_policy=_read_policy,
    )


def _runtime_collaborators() -> dependency_wiring.RuntimeCollaborators:
    return dependency_wiring.RuntimeCollaborators(
        index_folder=index_folder,
        evaluate_localization_batch=evaluate_localization_batch,
        load_tests=load_tests,
        format_tests_for_prompt=format_tests_for_prompt,
        build_frontier_context=build_frontier_context,
        format_frontier_context=format_frontier_context,
        compute_diff=compute_diff,
        format_diff=format_diff,
        interpret_diff=interpret_diff,
        generate_policy=generate_policy,
        emit_score=emit_score,
        classify_results=classify_results,
        read_policy=_read_policy,
        write_policy=_write_policy,
        get_writer_model=get_writer_model,
        start_observation=start_observation,
        find_repo_root=find_repo_root,
        default_pipeline=default_pipeline,
    )


def _build_dependencies() -> LoopDependencies:
    return dependency_wiring.build_loop_dependencies(
        _runtime_collaborators(),
        evaluate_policy_on_item=evaluate_policy_on_item,
        build_iteration_feedback=build_iteration_feedback,
        attempt_policy_generation=attempt_policy_generation,
    )


def _fallback_history(
    opt_model: OptimizationMachineModel,
    parent_trace: SpanHandle | None,
) -> list[IterationRecord]:
    repo_root = find_repo_root()
    pipeline = default_pipeline()
    fallback_parent = opt_model.context.run_trace or parent_trace
    pipeline_results, pipeline_success = run_policy_pipeline(
        pipeline,
        repo_root,
        repair_obs=fallback_parent,
        allow_no_parent=fallback_parent is None,
    )
    classified = classify_results(pipeline_results)
    failed_step = dependency_wiring.first_failed_step(pipeline_results)
    metrics_map: dict[str, float | int | bool | None] = {}
    return [
        IterationRecord(
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
    ]


def _baseline_snapshot_model(
    baseline_snapshot: "BaselineSnapshot | Mapping[str, object] | None",
) -> "BaselineSnapshot | None":
    if isinstance(baseline_snapshot, Mapping):
        from harness.telemetry.hosted.baselines import BaselineSnapshot as BaselineSnapshotModel

        return BaselineSnapshotModel.model_validate(baseline_snapshot)
    return baseline_snapshot


def run_loop(
    task: LCATask,
    iterations: int = 5,
    parent_trace: SpanHandle | None = None,
    baseline_snapshot: "BaselineSnapshot | Mapping[str, object] | None" = None,
    session_id: str | None = None,
) -> list[IterationRecord]:
    """
    Closed-loop optimizer: execute, score, rewrite policy, repeat.
    """
    root_meta = RunMetadata(task=task, iterations=iterations, session_id=session_id)
    opt_model: OptimizationMachineModel | None = None
    history: list[IterationRecord] = []
    with start_observation(
        name="run_loop",
        parent=parent_trace,
        input=task.model_dump(),
        metadata=root_meta.model_dump(),
    ) as run_span:
        with propagate_context(
            trace_name="run_loop",
            session_id=session_id,
            metadata={"iterations": iterations, "session_id": session_id},
        ):
            context = LoopContext(
                task=task,
                iterations=iterations,
                session_id=session_id,
                parent_trace=parent_trace,
                run_metadata=root_meta,
                baseline_snapshot=_baseline_snapshot_model(baseline_snapshot),
                run_trace=run_span,
            )
            opt_model = OptimizationMachineModel(
                context=context,
                deps=_build_dependencies(),
                max_policy_repairs=_MAX_POLICY_REPAIRS,
            )
            machine = OptimizationStateMachine(opt_model)
            try:
                history = machine.run()
            finally:
                flush_langfuse()
    if not history and opt_model is not None:
        history = _fallback_history(opt_model, parent_trace)
    return history
