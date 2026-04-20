from __future__ import annotations

from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory
from harness.localization.materialization.worktree import RepoMaterializationResult
from harness.localization.materialization.worktree import WorktreeManager
from harness.localization.models import (
    LCATask,
    LocalizationEvidence,
    LocalizationPrediction,
    LocalizationRunResult,
)
from harness.localization.runtime.execute import run_localization_task
from harness.scoring import BatchScoreSummary, ScoreBundle, summarize_batch_scores
from harness.scoring.token_usage import TokenUsageRecord
from harness.telemetry.observability_models import RunSummary, TaskRunSummary

LocalizationRunner = Callable[[LCATask, str, object | None], LocalizationRunResult]

# Allows tests or callers to inject a deterministic runner without touching core wiring.
DEFAULT_LOCALIZATION_RUNNER: LocalizationRunner | None = None


def _build_failure(
    category: LocalizationFailureCategory,
    message: str,
    task: LCATask | None,
    observability_summary: TaskRunSummary | None = None,
) -> LocalizationEvaluationFailure:
    return LocalizationEvaluationFailure(
        category=category,
        message=message,
        task_id=task.task_id if task else None,
        observability_summary=observability_summary,
    )


class LocalizationEvaluationFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: LocalizationFailureCategory
    message: str
    task_id: str | None = None
    observability_summary: TaskRunSummary | None = None


class LocalizationEvaluationTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: LCATask
    score_bundle: ScoreBundle
    prediction: LocalizationPrediction
    evidence: LocalizationEvidence | None = None
    materialization: RepoMaterializationResult | None = None
    trace_id: str | None = None
    token_usage: TokenUsageRecord | None = None
    observability_summary: TaskRunSummary | None = None


class LocalizationEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_summary: BatchScoreSummary | None = None
    machine_score: float | None = None
    items: list[LocalizationEvaluationTaskResult] = Field(default_factory=list)
    failure: LocalizationEvaluationFailure | None = None
    run_summary: RunSummary | None = None


def prediction_filenames(prediction: LocalizationPrediction) -> list[str]:
    """
    Explicit boundary for consumers that expose only filenames outside the
    semantic prediction stage.
    """
    return list(prediction.predicted_files)


def evaluate_localization_batch(
    tasks: Sequence[LCATask],
    *,
    dataset_provenance: str | None = None,
    worktree_manager: WorktreeManager | None = None,
    parent_trace: object | None = None,
    runner: LocalizationRunner | None = None,
) -> LocalizationEvaluationResult:
    """
    Shared localization evaluation backend used by the optimization/repair machine and hosted wrappers.
    Executes each task via run_localization_task, aggregates score bundles, and returns typed results/failures.
    """
    resolved_runner = runner or DEFAULT_LOCALIZATION_RUNNER
    task_results: list[LocalizationEvaluationTaskResult] = []
    backend_name = "ic"
    for task in tasks:
        try:
            run_result = run_localization_task(
                task,
                dataset_provenance=dataset_provenance,
                worktree_manager=worktree_manager,
                parent_trace=parent_trace,
                runner=resolved_runner,
            )
            if len(run_result) < 4:
                raise LocalizationEvaluationError(LocalizationFailureCategory.UNKNOWN, "Runner returned insufficient values", task_id=task.task_id)
            prediction, score_bundle, evidence, materialization = run_result[:4]
            usage = run_result[4] if len(run_result) > 4 else None
            runner_summary = getattr(run_result[5], "observability_summary", None) if len(run_result) > 5 else None
            if runner_summary is not None:
                backend_name = runner_summary.backend
            task_results.append(
                LocalizationEvaluationTaskResult(
                    task=task,
                    score_bundle=score_bundle,
                    prediction=prediction,
                    evidence=evidence,
                    materialization=materialization,
                    trace_id=getattr(parent_trace, "id", None),
                    token_usage=usage,
                    observability_summary=runner_summary,
                )
            )
        except LocalizationEvaluationError as exc:
            exc_summary = getattr(exc, "observability_summary", None)
            if exc_summary is not None:
                backend_name = exc_summary.backend
            summaries = [
                r.observability_summary
                for r in task_results
                if r.observability_summary is not None
            ]
            if exc_summary is not None:
                summaries.append(exc_summary)
            failure = _build_failure(exc.category, str(exc), task, exc_summary)
            return LocalizationEvaluationResult(failure=failure, items=task_results, run_summary=RunSummary.from_task_summaries(backend=backend_name, selected_tasks_count=len(tasks), task_summaries=summaries))
        except Exception as exc:  # noqa: BLE001
            failure = _build_failure(LocalizationFailureCategory.UNKNOWN, str(exc), task)
            return LocalizationEvaluationResult(failure=failure, items=task_results, run_summary=RunSummary.from_task_summaries(backend=backend_name, selected_tasks_count=len(tasks), task_summaries=[r.observability_summary for r in task_results if r.observability_summary is not None]))

    try:
        score_summary = summarize_batch_scores(
            {res.task.task_id: res.score_bundle for res in task_results}
        )
        return LocalizationEvaluationResult(
            score_summary=score_summary,
            machine_score=score_summary.aggregate_composed_score,
            items=task_results,
            failure=None,
            run_summary=RunSummary.from_task_summaries(
                backend=backend_name,
                selected_tasks_count=len(tasks),
                task_summaries=[
                    res.observability_summary
                    for res in task_results
                    if res.observability_summary is not None
                ],
            ),
        )
    except LocalizationEvaluationError as exc:
        failure = _build_failure(exc.category, str(exc), None)
        return LocalizationEvaluationResult(failure=failure, items=task_results)
    except Exception as exc:  # noqa: BLE001
        failure = _build_failure(LocalizationFailureCategory.SCORING, str(exc), None)
        return LocalizationEvaluationResult(failure=failure, items=task_results)
