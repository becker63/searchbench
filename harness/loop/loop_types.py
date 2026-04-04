from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from harness.pipeline.types import PipelineClassification, StepResult

if TYPE_CHECKING:
    from harness.observability.baselines import BaselineSnapshot


MetricValue = int | float | bool
MetricsMap = Mapping[str, object]


class TaskPayload(BaseModel):
    """Normalized task payload passed into loop/runner agents."""

    model_config = ConfigDict(extra="allow")

    symbol: str
    repo: str


class PreparedTasks(BaseModel):
    """Resolved IC/JC tasks for a single iteration."""

    model_config = ConfigDict(extra="forbid")

    base_task: TaskPayload
    resolved_repo_path: str | None
    jc_repo_id: str | None

    def build_iteration_tasks(self) -> tuple[dict[str, object], dict[str, object]]:
        base = self.base_task.model_dump()
        ic_task = dict(base)
        jc_task = dict(base)
        if self.resolved_repo_path is not None:
            ic_task["repo"] = self.resolved_repo_path
        if self.jc_repo_id is not None:
            jc_task["repo"] = self.jc_repo_id
        elif self.resolved_repo_path is not None:
            jc_task["repo"] = self.resolved_repo_path
        return ic_task, jc_task


class EvaluationResult(BaseModel):
    """Outputs from evaluating the currently accepted policy."""

    model_config = ConfigDict(extra="forbid")

    metrics: MetricsMap
    ic_result: dict[str, object]
    jc_result: dict[str, object]
    jc_metrics: MetricsMap
    comparison_summary: str | None
    policy_code: str
    success: bool = True
    error: str | None = None


class FeedbackPackage(BaseModel):
    """Inputs used to steer policy repair/generation."""

    model_config = ConfigDict(extra="forbid")

    tests: str
    scoring_context: str
    comparison_summary: str | None
    feedback: dict[str, str | float | int | bool | None]
    feedback_str: str
    guidance_hint: str
    diff_str: str
    diff_hint: str


class RepairOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: str
    pipeline_results: list[StepResult]
    success: bool
    attempts_used: int
    pipeline_feedback: PipelineClassification | None = None
    writer_error: str | None = None
    error: str | None = None
    failed_step: str | None = None
    failed_exit_code: int | None = None
    failed_summary: str | None = None
    accepted_policy_source: str | None = None
    accepted_policy_changed_by_pipeline: bool | None = None
    repair_attempts_reported: int | None = None
    max_policy_repairs: int | None = None


class AcceptedPolicy(BaseModel):
    """Represents the accepted policy artifact and provenance."""

    model_config = ConfigDict(extra="forbid")

    code: str
    source: str | None
    changed_by_pipeline: bool | None
    repair_attempts: int
    max_policy_repairs: int


class IterationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int
    metrics: MetricsMap
    pipeline_passed: bool | None = None
    pipeline_feedback: PipelineClassification | None = None
    failed_step: str | None = None
    failed_exit_code: int | None = None
    failed_summary: str | None = None
    accepted_policy: AcceptedPolicy | None = None
    error: str | None = None
    writer_error: str | None = None
    repair_attempts: int | None = None
    max_policy_repairs: int | None = None


class LoopContext(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task: TaskPayload
    iterations: int
    session_id: str | None = None
    parent_trace: object | None = None
    run_metadata: Mapping[str, object] | None = None
    baseline_snapshot: "BaselineSnapshot | None" = None
    run_trace: object | None
    history: list[IterationRecord] = Field(default_factory=list)
    prepared_tasks: PreparedTasks | None = None
    prev_score: float | None = None
    prev_classified: PipelineClassification | None = None
    last_classified: PipelineClassification | None = None
    current_iteration: int = 0


class LoopDependencies(BaseModel):
    """Functions injected for side effects to keep the state machine thin."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prepare_iteration_tasks: Callable[[TaskPayload, object | None], PreparedTasks]
    evaluate_policy_on_item: Callable[[dict[str, object], "BaselineSnapshot | None", object | None, int | None], EvaluationResult]
    build_iteration_feedback: Callable[[EvaluationResult, float | None, PipelineClassification | None, Path], FeedbackPackage]
    attempt_policy_generation: Callable[..., tuple[str | None, str | None]]
    run_policy_pipeline: Callable[..., tuple[list[StepResult], bool]]
    finalize_successful_policy: Callable[..., tuple[str, "AcceptedPolicyMeta"]]
    finalize_failed_repair: Callable[..., "FailedRepairDetails"]
    classify_results: Callable[[list[StepResult]], PipelineClassification]
    format_pipeline_failure_context: Callable[[list[StepResult], PipelineClassification | None, str | None, str | None, int], str]
    read_policy: Callable[[], str]
    write_policy: Callable[[str], None]
    get_writer_model: Callable[[], str | None]
    start_observation: Callable[..., object | None]
    find_repo_root: Callable[[], Path]
    default_pipeline: Callable[[], object]


class RepairContext(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    repo_root: Path
    evaluation: EvaluationResult
    feedback: FeedbackPackage
    max_repair_attempts: int
    parent_trace: object | None
    writer_model: str | None
    failure_context: str | None = None
    last_classified: PipelineClassification | None = None
    attempts_used: int = 0
    pipeline_results: list[StepResult] = Field(default_factory=list)
    success: bool = False
    pipeline_passed: bool = False
    candidate_code: str | None = None
    pipeline: object | None = None
    repair_observation: object | None = None
    writer_error: str | None = None
    pipeline_feedback: PipelineClassification | None = None
    failed_step: str | None = None
    failed_exit_code: int | None = None
    failed_summary: str | None = None
    accepted_policy_source: str | None = None
    accepted_policy_changed_by_pipeline: bool | None = None
    repair_attempts_reported: int | None = None
    max_policy_repairs_reported: int | None = None
    error: str | None = None


class AcceptedPolicyMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted_policy_source: str | None = None
    accepted_policy_changed_by_pipeline: bool | None = None
    repair_attempts: int
    max_policy_repairs: int


class FailedRepairDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_context: str
    policy_code: str
    failed_step: str | None = None
    failed_exit_code: int | None = None
    failed_summary: str | None = None
    error: str | None = None


def iteration_record_to_public_dict(record: IterationRecord) -> dict[str, object]:
    """Serialize iteration records for external consumers (CLI, scoring)."""
    data: dict[str, object] = dict(record.metrics)
    if record.pipeline_passed is not None:
        data["pipeline_passed"] = record.pipeline_passed
    if record.pipeline_feedback is not None:
        data["pipeline_feedback"] = record.pipeline_feedback.model_dump()
    if record.failed_step is not None:
        data["failed_step"] = record.failed_step
    if record.failed_exit_code is not None:
        data["failed_exit_code"] = record.failed_exit_code
    if record.failed_summary is not None:
        data["failed_summary"] = record.failed_summary
    if record.accepted_policy:
        data["accepted_policy_source"] = record.accepted_policy.source
        data["accepted_policy_changed_by_pipeline"] = record.accepted_policy.changed_by_pipeline
        data["repair_attempts"] = record.accepted_policy.repair_attempts
        data["max_policy_repairs"] = record.accepted_policy.max_policy_repairs
    if record.error is not None:
        data["error"] = record.error
    if record.writer_error is not None:
        data["writer_error"] = record.writer_error
    if record.repair_attempts is not None:
        data.setdefault("repair_attempts", record.repair_attempts)
    if record.max_policy_repairs is not None:
        data.setdefault("max_policy_repairs", record.max_policy_repairs)
    return data


class RepairMachineModel(BaseModel):
    """Mutable state model for the repair state machine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context: RepairContext
    deps: "LoopDependencies"
    state: str | None = None


class OptimizationMachineModel(BaseModel):
    """Mutable state model for the optimization state machine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context: LoopContext
    deps: "LoopDependencies"
    max_policy_repairs: int
    state: str | None = None
    prep_ok: bool = False
    current_iteration_span: object | None = None
    current_evaluation: EvaluationResult | None = None
    current_repair_outcome: RepairOutcome | None = None


def _rebuild_forward_refs() -> None:
    try:
        from harness.observability.baselines import BaselineSnapshot  # local import to avoid cycles
    except Exception:
        return
    LoopContext.model_rebuild(_types_namespace={"BaselineSnapshot": BaselineSnapshot})
    LoopDependencies.model_rebuild(_types_namespace={"BaselineSnapshot": BaselineSnapshot})
    RepairContext.model_rebuild(_types_namespace={"BaselineSnapshot": BaselineSnapshot})
    RepairMachineModel.model_rebuild(_types_namespace={"LoopDependencies": LoopDependencies})
    OptimizationMachineModel.model_rebuild(_types_namespace={"LoopDependencies": LoopDependencies})


_rebuild_forward_refs()
