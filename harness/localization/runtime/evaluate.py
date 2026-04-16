from __future__ import annotations

from typing import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory
from harness.localization.materialization.materialize import RepoMaterializer, RepoMaterializationResult
from harness.localization.models import LCATask, LocalizationEvidence, normalize_lca_task
from harness.localization.runtime.execute import run_localization_task
from harness.scoring import BatchScoreSummary, ScoreBundle, summarize_batch_scores
from harness.scoring.token_usage import TokenUsageRecord

LocalizationRunner = Callable[[LCATask, str, object | None], tuple[list[str], Mapping[str, object] | None]]

# Allows tests or callers to inject a deterministic runner without touching core wiring.
DEFAULT_LOCALIZATION_RUNNER: LocalizationRunner | None = None


def _build_failure(category: LocalizationFailureCategory, message: str, task: LCATask | None) -> LocalizationEvaluationFailure:
    return LocalizationEvaluationFailure(category=category, message=message, task_id=task.task_id if task else None)


class LocalizationEvaluationFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: LocalizationFailureCategory
    message: str
    task_id: str | None = None


class LocalizationEvaluationTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: LCATask
    score_bundle: ScoreBundle
    prediction: list[str] = Field(default_factory=list)
    evidence: LocalizationEvidence | None = None
    materialization: RepoMaterializationResult | None = None
    trace_id: str | None = None
    token_usage: TokenUsageRecord | None = None


class LocalizationEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_summary: BatchScoreSummary | None = None
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
) -> LocalizationEvaluationResult:
    """
    Shared localization evaluation backend used by the optimization/repair machine and hosted wrappers.
    Executes each task via run_localization_task, aggregates score bundles, and returns typed results/failures.
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
            prediction, score_bundle, evidence, materialization = run_result[:4]
            usage = run_result[4] if len(run_result) > 4 else None
            task_results.append(
                LocalizationEvaluationTaskResult(
                    task=task,
                    score_bundle=score_bundle,
                    prediction=list(prediction.predicted_files),
                    evidence=evidence,
                    materialization=materialization,
                    trace_id=getattr(parent_trace, "id", None),
                    token_usage=usage,
                )
            )
        except LocalizationEvaluationError as exc:
            failure = _build_failure(exc.category, str(exc), task)
            return LocalizationEvaluationResult(failure=failure, items=task_results)
        except Exception as exc:  # noqa: BLE001
            failure = _build_failure(LocalizationFailureCategory.UNKNOWN, str(exc), task)
            return LocalizationEvaluationResult(failure=failure, items=task_results)

    try:
        score_summary = summarize_batch_scores(
            {res.task.task_id: res.score_bundle for res in task_results}
        )
        return LocalizationEvaluationResult(
            score_summary=score_summary,
            machine_score=score_summary.aggregate_composed_score,
            items=task_results,
            failure=None,
        )
    except LocalizationEvaluationError as exc:
        failure = _build_failure(exc.category, str(exc), None)
        return LocalizationEvaluationResult(failure=failure, items=task_results)
    except Exception as exc:  # noqa: BLE001
        failure = _build_failure(LocalizationFailureCategory.SCORING, str(exc), None)
        return LocalizationEvaluationResult(failure=failure, items=task_results)
