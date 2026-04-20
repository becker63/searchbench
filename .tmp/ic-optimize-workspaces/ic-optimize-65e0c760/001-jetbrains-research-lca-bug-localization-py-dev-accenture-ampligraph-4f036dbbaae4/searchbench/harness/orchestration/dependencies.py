from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from harness.localization.materialization.worktree import WorktreeManager
from harness.localization.models import LCATask
from harness.localization.runtime.evaluate import (
    LocalizationEvaluationResult,
    LocalizationRunner,
)
from harness.pipeline.types import PipelineClassification, StepResult
from harness.prompts import WriterOptimizationBrief
from harness.utils.diff import DiffSummary
from harness.utils.type_loader import FrontierContext
from harness.utils.warnings import suppress_third_party_invalid_escape_warnings
from .types import (
    AcceptedPolicyMeta,
    AttemptPolicyGeneration,
    BuildIterationFeedback,
    EvaluatePolicyOnItem,
    FailedRepairDetails,
    LoopDependencies,
    PreparedTasks,
    SpanHandle,
    StartObservation,
    SupportsPipelineRun,
    format_failure_context_payload,
)


IndexFolder = Callable[[str], Mapping[str, object]]
ReadPolicy = Callable[[], str]
WritePolicy = Callable[[str], None]
GetWriterModel = Callable[[], str | None]
FindRepoRoot = Callable[[], Path]
PipelineFactory = Callable[[], SupportsPipelineRun]
ClassifyResults = Callable[[list[StepResult]], PipelineClassification]
TestSpecs = list[tuple[str, str]]
LoadTests = Callable[[Path], TestSpecs]
FormatTestsForPrompt = Callable[[TestSpecs], str]
BuildFrontierContext = Callable[[Path], FrontierContext]
FormatFrontierContext = Callable[[FrontierContext], str]
FormatDiff = Callable[[DiffSummary], str]
InterpretDiff = Callable[[DiffSummary], str]


class EvaluateLocalizationBatch(Protocol):
    def __call__(
        self,
        tasks: Sequence[LCATask],
        *,
        dataset_provenance: str | None = None,
        worktree_manager: WorktreeManager | None = None,
        parent_trace: object | None = None,
        runner: LocalizationRunner | None = None,
    ) -> LocalizationEvaluationResult: ...


class PolicyGenerator(Protocol):
    def __call__(
        self,
        current_policy: str,
        tests: str,
        frontier_context: str,
        frontier_context_details: FrontierContext,
        optimization_brief: WriterOptimizationBrief,
        feedback: dict[str, object] | None = None,
        parent_trace: object | None = None,
        failure_context: str | None = None,
        repair_attempt: int = 0,
    ) -> str: ...


class EmitScore(Protocol):
    def __call__(
        self,
        *,
        name: str,
        value: float | int | bool | str,
        data_type: str | None = None,
        trace_id: str | None = None,
        observation_id: str | None = None,
        session_id: str | None = None,
        config_id: str | None = None,
        comment: str | None = None,
        score_id: str | None = None,
    ) -> None: ...


class ComputeRawDiff(Protocol):
    def __call__(
        self,
        prev_score: float,
        curr_score: float,
        prev_classified: object,
        curr_classified: object,
    ) -> DiffSummary: ...


@dataclass(frozen=True)
class RuntimeCollaborators:
    index_folder: IndexFolder
    evaluate_localization_batch: EvaluateLocalizationBatch
    load_tests: LoadTests
    format_tests_for_prompt: FormatTestsForPrompt
    build_frontier_context: BuildFrontierContext
    format_frontier_context: FormatFrontierContext
    compute_diff: ComputeRawDiff
    format_diff: FormatDiff
    interpret_diff: InterpretDiff
    generate_policy: PolicyGenerator
    emit_score: EmitScore
    classify_results: ClassifyResults
    read_policy: ReadPolicy
    write_policy: WritePolicy
    get_writer_model: GetWriterModel
    start_observation: StartObservation
    find_repo_root: FindRepoRoot
    default_pipeline: PipelineFactory


def prepare_iteration_tasks(
    task: LCATask,
    parent_trace: SpanHandle | None,
    *,
    start_observation: StartObservation,
    index_folder: IndexFolder,
) -> PreparedTasks:
    resolved_repo_path = task.repo
    jc_repo_id: str | None = None
    if isinstance(task.repo, str):
        with start_observation(
            name="index_repo",
            parent=parent_trace,
            metadata={"repo": task.repo},
        ) as index_obs:
            with suppress_third_party_invalid_escape_warnings(task.repo):
                result = index_folder(task.repo)
            repo_id = str(result.get("repo", ""))
            jc_repo_id = repo_id
            try:
                index_obs.update(metadata={"repo": repo_id})
            except Exception:
                pass
    if resolved_repo_path is None:
        raise RuntimeError("Task missing repo")
    return PreparedTasks(
        task=task.with_repo(resolved_repo_path),
        resolved_repo_path=resolved_repo_path,
        jc_repo_id=jc_repo_id,
    )


def compute_pipeline_diff(
    prev_score: float,
    current_score: float,
    prev_classified: PipelineClassification,
    current_classified: PipelineClassification,
    *,
    compute_diff: ComputeRawDiff,
) -> DiffSummary:
    return compute_diff(
        prev_score,
        current_score,
        prev_classified.model_dump(),
        current_classified.model_dump(),
    )


def step_succeeded(step: StepResult) -> bool:
    return step.exit_code == 0


def first_failed_step(steps: list[StepResult]) -> StepResult | None:
    for step in steps:
        if not step_succeeded(step):
            return step
    return None


def run_policy_pipeline(
    pipeline: SupportsPipelineRun,
    repo_root: Path,
    repair_obs: SpanHandle | None = None,
    allow_no_parent: bool = False,
    *,
    emit_score: EmitScore,
) -> tuple[list[StepResult], bool]:
    try:
        raw_results = list(
            pipeline.run(
                repo_root,
                observation=repair_obs,
                allow_no_parent=allow_no_parent,
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
    if not any([trace_id, observation_id, session_id_val]):
        return step_results, all(step_succeeded(result) for result in step_results)

    for step in step_results:
        score_id = (
            f"{observation_id or trace_id}-pipeline.{step.name}.exit_code"
            if (observation_id or trace_id)
            else None
        )
        emit_score(
            name=f"pipeline.{step.name}.exit_code",
            value=step.exit_code,
            data_type="NUMERIC",
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
        emit_score(
            name=f"pipeline.{step.name}.passed",
            value=int(step.success),
            data_type="BOOLEAN",
            trace_id=trace_id,
            observation_id=observation_id,
            session_id=session_id_val,
            score_id=pass_score_id,
        )
    return step_results, all(step_succeeded(result) for result in step_results)


def finalize_successful_policy(
    *,
    new_code: str,
    attempts_used: int,
    max_repair_attempts: int,
    repair_obs: object | None = None,
    read_policy: ReadPolicy,
) -> tuple[str, AcceptedPolicyMeta]:
    del repair_obs
    final_policy = read_policy()
    meta = AcceptedPolicyMeta(
        accepted_policy_source="post_pipeline_disk",
        accepted_policy_changed_by_pipeline=final_policy != new_code,
        repair_attempts=attempts_used - 1,
        max_policy_repairs=max_repair_attempts,
    )
    return final_policy, meta


def format_pipeline_failure_context(
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


def finalize_failed_repair(
    *,
    pipeline_results: list[StepResult],
    classified: PipelineClassification,
    writer_model: str | None,
    attempts_used: int,
    writer_error: str | None,
    read_policy: ReadPolicy,
) -> FailedRepairDetails:
    failed_step = first_failed_step(pipeline_results)
    failed_summary = None
    if failed_step:
        stderr = failed_step.stderr
        stdout = failed_step.stdout
        text = stderr.strip() if stderr.strip() else stdout.strip()
        failed_summary = text[:400] if text else None
    failure_context = format_pipeline_failure_context(
        pipeline_results,
        classified,
        writer_error,
        writer_model,
        attempts_used,
    )
    return FailedRepairDetails(
        failure_context=failure_context,
        policy_code=read_policy(),
        failed_step=failed_step.name if failed_step else None,
        failed_exit_code=failed_step.exit_code if failed_step else None,
        failed_summary=failed_summary,
        error="pipeline_failed",
    )


def build_loop_dependencies(
    collaborators: RuntimeCollaborators,
    *,
    evaluate_policy_on_item: EvaluatePolicyOnItem,
    build_iteration_feedback: BuildIterationFeedback,
    attempt_policy_generation: AttemptPolicyGeneration,
) -> LoopDependencies:
    def prepare_tasks(task: LCATask, parent_trace: SpanHandle | None) -> PreparedTasks:
        return prepare_iteration_tasks(
            task,
            parent_trace,
            start_observation=collaborators.start_observation,
            index_folder=collaborators.index_folder,
        )

    def run_pipeline(
        pipeline: SupportsPipelineRun,
        repo_root: Path,
        repair_obs: SpanHandle | None = None,
        allow_no_parent: bool = False,
    ) -> tuple[list[StepResult], bool]:
        return run_policy_pipeline(
            pipeline,
            repo_root,
            repair_obs=repair_obs,
            allow_no_parent=allow_no_parent,
            emit_score=collaborators.emit_score,
        )

    def finalize_success(
        *,
        new_code: str,
        attempts_used: int,
        max_repair_attempts: int,
        repair_obs: object | None = None,
    ) -> tuple[str, AcceptedPolicyMeta]:
        return finalize_successful_policy(
            new_code=new_code,
            attempts_used=attempts_used,
            max_repair_attempts=max_repair_attempts,
            repair_obs=repair_obs,
            read_policy=collaborators.read_policy,
        )

    def finalize_failure(
        *,
        pipeline_results: list[StepResult],
        classified: PipelineClassification,
        writer_model: str | None,
        attempts_used: int,
        writer_error: str | None,
    ) -> FailedRepairDetails:
        return finalize_failed_repair(
            pipeline_results=pipeline_results,
            classified=classified,
            writer_model=writer_model,
            attempts_used=attempts_used,
            writer_error=writer_error,
            read_policy=collaborators.read_policy,
        )

    return LoopDependencies(
        prepare_iteration_tasks=prepare_tasks,
        evaluate_policy_on_item=evaluate_policy_on_item,
        build_iteration_feedback=build_iteration_feedback,
        attempt_policy_generation=attempt_policy_generation,
        run_policy_pipeline=run_pipeline,
        finalize_successful_policy=finalize_success,
        finalize_failed_repair=finalize_failure,
        classify_results=collaborators.classify_results,
        format_pipeline_failure_context=format_pipeline_failure_context,
        read_policy=collaborators.read_policy,
        write_policy=collaborators.write_policy,
        get_writer_model=collaborators.get_writer_model,
        start_observation=collaborators.start_observation,
        find_repo_root=collaborators.find_repo_root,
        default_pipeline=collaborators.default_pipeline,
    )
