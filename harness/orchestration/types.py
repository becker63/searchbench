from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, ContextManager, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.pipeline.types import PipelineClassification, StepResult
from harness.utils.type_loader import FrontierContext
from harness.localization.models import LCAContext, LCATaskIdentity

if TYPE_CHECKING:
    from harness.telemetry.hosted.baselines import BaselineSnapshot


MetricValue = int | float | bool
MetricsMap = Mapping[str, MetricValue | None]
ScalarMetrics = dict[str, MetricValue | None]
FeedbackMap = Mapping[str, str | float | int | bool | None]


@runtime_checkable
class SpanHandle(Protocol):
    def end(self, **kwargs: object) -> None: ...

    def update(self, **kwargs: object) -> None: ...


class MetricEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: MetricValue | None


class EvaluationMetrics(BaseModel):
    """Canonical metrics container with stable fields and extensible entries."""

    model_config = ConfigDict(extra="forbid")

    score: MetricValue | None = None
    iteration_regression: bool | None = None
    additional_metrics: list[MetricEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_mapping(cls, value: object) -> object:
        if isinstance(value, Mapping):
            score = value.get("score")
            iteration_regression = value.get("iteration_regression")
            additional_raw = value.get("additional_metrics", [])
            unknown_keys = [k for k in value.keys() if k not in {"score", "iteration_regression", "additional_metrics"}]
            if unknown_keys:
                raise ValueError(f"Unknown metric keys: {unknown_keys}")
            additional: list[dict[str, object]] = []
            if isinstance(additional_raw, list):
                additional = [
                    {"name": entry.get("name"), "value": entry.get("value")}
                    for entry in additional_raw
                    if isinstance(entry, Mapping)
                ]
            return {
                "score": score,
                "iteration_regression": iteration_regression,
                "additional_metrics": additional,
            }
        return value

    def as_map(self) -> dict[str, MetricValue | None]:
        data: dict[str, MetricValue | None] = {}
        if self.score is not None:
            data["score"] = self.score
        if self.iteration_regression is not None:
            data["iteration_regression"] = self.iteration_regression
        for entry in self.additional_metrics:
            data[entry.name] = entry.value
        return data

    def items(self):
        return self.as_map().items()

    def get(self, key: str, default: object | None = None) -> object | None:
        return self.as_map().get(key, default)  # type: ignore[return-value]


class ResultEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: object | None


class ICResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[ResultEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_mapping(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {"entries": [{"name": k, "value": v} for k, v in value.items()]}
        return value

    def as_map(self) -> dict[str, object]:
        return {entry.name: entry.value for entry in self.entries}

    def get(self, key: str, default: object | None = None) -> object | None:
        return self.as_map().get(key, default)


class JCResult(ICResult):
    """JC result container."""

    model_config = ConfigDict(extra="forbid")


class FeedbackEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: str | float | int | bool | None


class FeedbackEntries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[FeedbackEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_mapping(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {"entries": [{"name": k, "value": v} for k, v in value.items()]}
        return value

    def as_map(self) -> dict[str, str | float | int | bool | None]:
        return {entry.name: entry.value for entry in self.entries}

    def get(self, key: str, default: object | None = None) -> object | None:
        return self.as_map().get(key, default)

    def items(self):
        return self.as_map().items()


class TaskPayload(BaseModel):
    """Localization-native task payload passed into orchestration/evaluation agents."""

    model_config = ConfigDict(extra="forbid")

    identity: LCATaskIdentity
    context: LCAContext
    repo: str
    changed_files: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_mapping(cls, value: object) -> object:
        if isinstance(value, Mapping):
            identity_val = value.get("identity") or {}
            context_val = value.get("context") or {}
            return {
                "identity": identity_val,
                "context": context_val,
                "repo": value.get("repo"),
                "changed_files": value.get("changed_files", []),
            }
        return value


class ICTaskPayload(TaskPayload):
    """Iteration task payload for localization runs (IC backend)."""

    model_config = ConfigDict(extra="forbid")


class JCTaskPayload(TaskPayload):
    """Iteration task payload for localization runs (JC backend)."""

    model_config = ConfigDict(extra="forbid")


class RunMetadata(BaseModel):
    """Run-level metadata persisted for tracing and external visibility."""

    model_config = ConfigDict(extra="forbid")

    iterations: int
    task: TaskPayload
    session_id: str | None = None

class PreparedTasks(BaseModel):
    """Resolved IC/JC tasks for a single iteration."""

    model_config = ConfigDict(extra="forbid")

    base_task: TaskPayload
    resolved_repo_path: str | None
    jc_repo_id: str | None
    ic_task: ICTaskPayload | None = None
    jc_task: JCTaskPayload | None = None

    def _resolve_ic_repo(self) -> str:
        if self.resolved_repo_path is not None:
            return self.resolved_repo_path
        return self.base_task.repo

    def _resolve_jc_repo(self) -> str:
        if self.jc_repo_id is not None:
            return self.jc_repo_id
        if self.resolved_repo_path is not None:
            return self.resolved_repo_path
        return self.base_task.repo

    def to_ic_request(self) -> ICTaskPayload:
        if self.ic_task is not None:
            return self.ic_task
        return ICTaskPayload(
            identity=self.base_task.identity,
            context=self.base_task.context,
            repo=self._resolve_ic_repo(),
            changed_files=self.base_task.changed_files,
        )

    def to_jc_request(self) -> JCTaskPayload:
        if self.jc_task is not None:
            return self.jc_task
        return JCTaskPayload(
            identity=self.base_task.identity,
            context=self.base_task.context,
            repo=self._resolve_jc_repo(),
            changed_files=self.base_task.changed_files,
        )

    def build_iteration_tasks(self) -> "IterationTasks":
        return IterationTasks(ic_task=self.to_ic_request(), jc_task=self.to_jc_request())


class IterationTasks(BaseModel):
    """Concrete task payloads for an iteration."""

    model_config = ConfigDict(extra="forbid")

    ic_task: ICTaskPayload
    jc_task: JCTaskPayload


class EvaluationResult(BaseModel):
    """Outputs from evaluating the currently accepted policy."""

    model_config = ConfigDict(extra="forbid")

    metrics: EvaluationMetrics
    ic_result: ICResult
    jc_result: JCResult
    jc_metrics: EvaluationMetrics
    comparison_summary: str | None
    policy_code: str
    success: bool = True
    error: str | None = None
    control_score: float | None = None


class FeedbackPackage(BaseModel):
    """Inputs used to steer policy repair/generation."""

    model_config = ConfigDict(extra="forbid")

    tests: str
    frontier_context: str
    frontier_context_details: FrontierContext
    comparison_summary: str | None
    feedback: FeedbackEntries
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
    metrics: EvaluationMetrics
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
    parent_trace: SpanHandle | None = None
    run_metadata: RunMetadata | None = None
    baseline_snapshot: "BaselineSnapshot | None" = None
    run_trace: SpanHandle | None
    history: list[IterationRecord] = Field(default_factory=list)
    prepared_tasks: PreparedTasks | None = None
    prev_score: float | None = None
    prev_classified: PipelineClassification | None = None
    last_classified: PipelineClassification | None = None
    current_iteration: int = 0


class LoopDependencies(BaseModel):
    """Functions injected for side effects to keep the state machine thin."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prepare_iteration_tasks: Callable[[TaskPayload, SpanHandle | None], PreparedTasks]
    evaluate_policy_on_item: Callable[[ICTaskPayload, "BaselineSnapshot | None", SpanHandle | None, int | None], EvaluationResult]
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
    start_observation: Callable[..., ContextManager[SpanHandle]]
    find_repo_root: Callable[[], Path]
    default_pipeline: Callable[[], object]


class RepairContext(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    repo_root: Path
    evaluation: EvaluationResult
    feedback: FeedbackPackage
    max_repair_attempts: int
    parent_trace: SpanHandle | None
    writer_model: str | None
    failure_context: str | None = None
    last_classified: PipelineClassification | None = None
    attempts_used: int = 0
    pipeline_results: list[StepResult] = Field(default_factory=list)
    success: bool = False
    pipeline_passed: bool = False
    candidate_code: str | None = None
    pipeline: object | None = None
    repair_observation: SpanHandle | None = None
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
    repair_observation_cm: ContextManager[SpanHandle] | None = None


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
    data: dict[str, object] = {k: v for k, v in record.metrics.as_map().items()}
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


def format_failure_context_payload(
    pipeline_results: Sequence[StepResult],
    classified: PipelineClassification | None,
    writer_error: str | None,
    model_name: str | None,
    repair_attempt: int,
) -> str:
    """Centralized failure-context serialization for writer prompts."""
    from harness.utils.failure_context import pack_failure_context  # local import to avoid cycles

    pipeline_dump = [step.model_dump() for step in pipeline_results]
    classified_dump = classified.model_dump() if classified else None
    failed_step = next((step for step in pipeline_results if step.exit_code != 0), None)
    failed_dump = failed_step.model_dump() if failed_step else None
    return pack_failure_context(
        pipeline_dump,
        classified_dump,
        failed_dump,
        writer_error,
        model_name,
        repair_attempt=repair_attempt,
    )


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
    current_iteration_span: SpanHandle | None = None
    current_iteration_cm: ContextManager[SpanHandle] | None = None
    current_evaluation: EvaluationResult | None = None
    current_repair_outcome: RepairOutcome | None = None


def _rebuild_forward_refs() -> None:
    try:
        from harness.telemetry.hosted.baselines import BaselineSnapshot  # local import to avoid cycles
    except Exception:
        BaselineSnapshot = object  # type: ignore[misc,assignment]
    LoopContext.model_rebuild(_types_namespace={"BaselineSnapshot": BaselineSnapshot, "RunMetadata": RunMetadata})
    LoopDependencies.model_rebuild(_types_namespace={"BaselineSnapshot": BaselineSnapshot})
    RepairContext.model_rebuild(_types_namespace={"BaselineSnapshot": BaselineSnapshot})
    RepairMachineModel.model_rebuild(_types_namespace={"LoopDependencies": LoopDependencies})
    OptimizationMachineModel.model_rebuild(_types_namespace={"LoopDependencies": LoopDependencies})


_rebuild_forward_refs()
