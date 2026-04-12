from __future__ import annotations

from enum import Enum
from statistics import mean
from typing import Callable, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from harness.localization.runtime.execute import run_localization_task
from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory
from harness.localization.materialization.materialize import RepoMaterializer, RepoMaterializationResult
from harness.localization.models import LCATask, LocalizationEvidence, LocalizationMetrics, normalize_lca_task
from harness.localization.token_usage import TokenUsageRecord

LocalizationRunner = Callable[[LCATask, str, object | None], tuple[list[str], Mapping[str, object] | None]]

# Allows tests or callers to inject a deterministic runner without touching core wiring.
DEFAULT_LOCALIZATION_RUNNER: LocalizationRunner | None = None


def _avg(values: Iterable[float]) -> float | None:
    collected = [v for v in values if isinstance(v, (float, int))]
    return mean(collected) if collected else None


def _build_failure(category: LocalizationFailureCategory, message: str, task: LCATask | None) -> LocalizationEvaluationFailure:
    return LocalizationEvaluationFailure(category=category, message=message, task_id=task.task_id if task else None)


class MachineScorePolicy(str, Enum):
    AGGREGATE = "aggregate"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    HIT = "hit"


class LocalizationEvaluationFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: LocalizationFailureCategory
    message: str
    task_id: str | None = None


class LocalizationEvaluationTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: LCATask
    metrics: LocalizationMetrics
    prediction: list[str] = Field(default_factory=list)
    evidence: LocalizationEvidence | None = None
    materialization: RepoMaterializationResult | None = None
    trace_id: str | None = None
    token_usage: TokenUsageRecord | None = None


class LocalizationEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregate_score: float | None
    aggregate_metrics: LocalizationMetrics | None = None
    machine_score: float | None = None
    items: list[LocalizationEvaluationTaskResult] = Field(default_factory=list)
    failure: LocalizationEvaluationFailure | None = None


def evaluate_localization_batch(
    tasks: Sequence[LCATask | Mapping[str, object]],
    *,
    dataset_source: str | None = None,
    materializer: RepoMaterializer | None = None,
    parent_trace: object | None = None,
    runner: LocalizationRunner | None = None,
    machine_score_policy: MachineScorePolicy = MachineScorePolicy.AGGREGATE,
) -> LocalizationEvaluationResult:
    """
    Shared localization evaluation backend used by the optimization/repair machine and hosted wrappers.
    Executes each task via run_localization_task, aggregates scores, and returns typed results/failures.
    """
    normalized_tasks = [t if isinstance(t, LCATask) else normalize_lca_task(t) for t in tasks]
    resolved_runner = runner or DEFAULT_LOCALIZATION_RUNNER
    task_results: list[LocalizationEvaluationTaskResult] = []
    for task in normalized_tasks:
        try:
            run_result = run_localization_task(
                task,
                dataset_source=dataset_source,
                materializer=materializer,
                parent_trace=parent_trace,
                runner=resolved_runner,
            )
            if len(run_result) < 4:
                raise LocalizationEvaluationError(LocalizationFailureCategory.UNKNOWN, "Runner returned insufficient values", task_id=task.task_id)
            prediction, metrics, evidence, materialization = run_result[:4]
            usage = run_result[4] if len(run_result) > 4 else None
            task_results.append(
                LocalizationEvaluationTaskResult(
                    task=task,
                    metrics=metrics,
                    prediction=list(prediction.predicted_files),
                    evidence=evidence,
                    materialization=materialization,
                    trace_id=getattr(parent_trace, "id", None),
                    token_usage=usage,
                )
            )
        except LocalizationEvaluationError as exc:
            failure = _build_failure(exc.category, str(exc), task)
            return LocalizationEvaluationResult(aggregate_score=None, failure=failure, items=task_results)
        except Exception as exc:  # noqa: BLE001
            failure = _build_failure(LocalizationFailureCategory.UNKNOWN, str(exc), task)
            return LocalizationEvaluationResult(aggregate_score=None, failure=failure, items=task_results)

    try:
        aggregate_score = _avg([res.metrics.score for res in task_results])
        aggregate_metrics = None
        if task_results and aggregate_score is not None:
            aggregate_metrics = LocalizationMetrics(
                precision=_avg(res.metrics.precision for res in task_results) or 0.0,
                recall=_avg(res.metrics.recall for res in task_results) or 0.0,
                f1=_avg(res.metrics.f1 for res in task_results) or aggregate_score,
                hit=_avg(res.metrics.hit for res in task_results) or 0.0,
                score=aggregate_score,
            )
        resolved_policy = machine_score_policy or MachineScorePolicy.AGGREGATE
        machine_score = _derive_machine_score(
            aggregate_score=aggregate_score,
            aggregate_metrics=aggregate_metrics,
            policy=resolved_policy,
        )
        return LocalizationEvaluationResult(
            aggregate_score=aggregate_score,
            aggregate_metrics=aggregate_metrics,
            machine_score=machine_score,
            items=task_results,
            failure=None,
        )
    except LocalizationEvaluationError as exc:
        failure = _build_failure(exc.category, str(exc), None)
        return LocalizationEvaluationResult(aggregate_score=None, failure=failure, items=task_results)
    except Exception as exc:  # noqa: BLE001
        failure = _build_failure(LocalizationFailureCategory.SCORING, str(exc), None)
        return LocalizationEvaluationResult(aggregate_score=None, failure=failure, items=task_results)


def _derive_machine_score(
    *,
    aggregate_score: float | None,
    aggregate_metrics: LocalizationMetrics | None,
    policy: MachineScorePolicy,
) -> float | None:
    if policy == MachineScorePolicy.AGGREGATE:
        return aggregate_score
    if aggregate_metrics is None:
        return aggregate_score
    if policy == MachineScorePolicy.PRECISION:
        return aggregate_metrics.precision
    if policy == MachineScorePolicy.RECALL:
        return aggregate_metrics.recall
    if policy == MachineScorePolicy.F1:
        return aggregate_metrics.f1
    if policy == MachineScorePolicy.HIT:
        return aggregate_metrics.hit
    raise LocalizationEvaluationError(LocalizationFailureCategory.SCORING, f"Unknown machine score policy: {policy}")
