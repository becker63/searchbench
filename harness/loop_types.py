from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .pipeline.types import StepResult

if TYPE_CHECKING:
    from .observability.baselines import BaselineSnapshot


class PreparedTasks(BaseModel):
    """Resolved IC/JC tasks for a single iteration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_task: Mapping[str, object]
    resolved_repo_path: str | None
    jc_repo_id: str | None

    def build_iteration_tasks(self) -> tuple[dict[str, object], dict[str, object]]:
        ic_task = dict(self.base_task)
        jc_task = dict(self.base_task)
        if self.resolved_repo_path is not None:
            ic_task["repo"] = self.resolved_repo_path
        if self.jc_repo_id is not None:
            jc_task["repo"] = self.jc_repo_id
        elif self.resolved_repo_path is not None:
            jc_task["repo"] = self.resolved_repo_path
        return ic_task, jc_task


class EvaluationResult(BaseModel):
    """Outputs from evaluating the currently accepted policy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: dict[str, object]
    ic_result: Mapping[str, object]
    jc_result: Mapping[str, object]
    jc_metrics: Mapping[str, object]
    comparison_summary: str | None
    policy_code: str
    success: bool = True
    error: str | None = None


class FeedbackPackage(BaseModel):
    """Inputs used to steer policy repair/generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tests: str
    scoring_context: str
    comparison_summary: str | None
    feedback: dict[str, object]
    feedback_str: str
    guidance_hint: str
    diff_str: str
    diff_hint: str


class PipelineStepResult(BaseModel):
    """Structured view of a pipeline step result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str

    @classmethod
    def from_mapping(cls, step: Mapping[str, object] | object) -> "PipelineStepResult":
        if isinstance(step, Mapping):
            name = str(step.get("name") or "")
            success = bool(step.get("success", False))
            exit_code = step.get("exit_code")
            exit_code_val = int(exit_code) if isinstance(exit_code, int) else None
            stdout = str(step.get("stdout") or "")
            stderr = str(step.get("stderr") or "")
            return cls(name=name, success=success, exit_code=exit_code_val, stdout=stdout, stderr=stderr)
        name = getattr(step, "name", "") or ""
        success = bool(getattr(step, "success", False))
        exit_code_val = getattr(step, "exit_code", None)
        exit_code_val = int(exit_code_val) if isinstance(exit_code_val, int) else None
        stdout = str(getattr(step, "stdout", "") or "")
        stderr = str(getattr(step, "stderr", "") or "")
        return cls(name=name, success=success, exit_code=exit_code_val, stdout=stdout, stderr=stderr)

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class RepairOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    policy: str
    pipeline_results: list[PipelineStepResult]
    success: bool
    attempts_used: int
    metadata: dict[str, object]


class AcceptedPolicy(BaseModel):
    """Represents the accepted policy artifact and provenance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str
    source: str | None
    changed_by_pipeline: bool | None
    repair_attempts: int
    max_policy_repairs: int


class IterationRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    iteration: int
    metrics: dict[str, object]
    pipeline_passed: bool | None = None
    pipeline_feedback: Mapping[str, object] | None = None
    failed_step: str | None = None
    failed_exit_code: int | None = None
    failed_summary: str | None = None
    accepted_policy: AcceptedPolicy | None = None
    error: str | None = None
    writer_error: str | None = None
    repair_attempts: int | None = None
    max_policy_repairs: int | None = None


class LoopContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: Mapping[str, object]
    iterations: int
    baseline_snapshot: "BaselineSnapshot | None" = None
    run_trace: object | None
    history: list[IterationRecord] = Field(default_factory=list)
    prepared_tasks: PreparedTasks | None = None
    prev_score: float | None = None
    prev_classified: dict[str, object] | None = None
    current_iteration: int = 0


class LoopDependencies(BaseModel):
    """Functions injected for side effects to keep the state machine thin."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prepare_iteration_tasks: Callable[[Mapping[str, object], object | None], PreparedTasks]
    evaluate_policy_on_item: Callable[[Mapping[str, object], "BaselineSnapshot | None", object | None, int | None], EvaluationResult]
    build_iteration_feedback: Callable[[EvaluationResult, float | None, dict[str, object] | None, Path], FeedbackPackage]
    attempt_policy_generation: Callable[..., tuple[str | None, str | None]]
    run_policy_pipeline: Callable[..., tuple[list[PipelineStepResult], list[StepResult], bool]]
    finalize_successful_policy: Callable[..., tuple[str, dict[str, object]]]
    finalize_failed_repair: Callable[..., tuple[str, str]]
    classify_results: Callable[[Sequence[StepResult]], Mapping[str, object]]
    format_pipeline_failure_context: Callable[[list[Mapping[str, object]], Mapping[str, object] | None, str | None, str | None, int], str]
    read_policy: Callable[[], str]
    write_policy: Callable[[str], None]
    get_writer_model: Callable[[], str | None]
    record_score: Callable[[object, str, float, Mapping[str, object] | None], None]
    start_span: Callable[..., object | None]
    find_repo_root: Callable[[], Path]
    default_pipeline: Callable[[], object]


class RepairContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_root: Path
    evaluation: EvaluationResult
    feedback: FeedbackPackage
    max_repair_attempts: int
    parent_trace: object | None
    writer_model: str | None
    failure_context: str | None = None
    last_classified: Mapping[str, object] | None = None
    attempts_used: int = 0
    metadata: dict[str, object] = Field(default_factory=dict)
    pipeline_results: list[PipelineStepResult] = Field(default_factory=list)
    success: bool = False
    pipeline_passed: bool = False
    candidate_code: str | None = None
    pipeline: object | None = None
    repair_observation: object | None = None


def iteration_record_to_public_dict(record: IterationRecord) -> dict[str, object]:
    """Serialize iteration records for external consumers (CLI, scoring)."""
    data: dict[str, object] = dict(record.metrics)
    if record.pipeline_passed is not None:
        data["pipeline_passed"] = record.pipeline_passed
    if record.pipeline_feedback is not None:
        data["pipeline_feedback"] = dict(record.pipeline_feedback)
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
    started: bool = False


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
        from .observability.baselines import BaselineSnapshot  # local import to avoid cycles
    except Exception:
        return
    LoopContext.model_rebuild(_types_namespace={"BaselineSnapshot": BaselineSnapshot})
    LoopDependencies.model_rebuild(_types_namespace={"BaselineSnapshot": BaselineSnapshot})
    RepairContext.model_rebuild(_types_namespace={"BaselineSnapshot": BaselineSnapshot})
    RepairMachineModel.model_rebuild(_types_namespace={"LoopDependencies": LoopDependencies})
    OptimizationMachineModel.model_rebuild(_types_namespace={"LoopDependencies": LoopDependencies})


_rebuild_forward_refs()
