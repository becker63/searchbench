from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.localization.errors import (
    LocalizationEvaluationError,
    LocalizationFailureCategory,
)
from harness.localization.materialization.worktree import RepoMaterializationResult
from harness.localization.materialization.worktree import WorktreeManager
from harness.localization.models import (
    LCATask,
    LocalizationEvidence,
    LocalizationGold,
    LocalizationPrediction,
    LocalizationRunResult,
)
from harness.localization.runtime.evaluate import (
    LocalizationEvaluationFailure,
    evaluate_localization_batch,
)
from harness.scoring import (
    AggregateComponentScore,
    BatchScoreSummary,
    summarize_task_score,
)
from harness.scoring.batch import TaskScoreSummary
from harness.localization.telemetry import build_localization_telemetry
from harness.scoring.token_usage import TokenUsageRecord
from harness.telemetry.tracing import propagate_context, start_observation
from harness.telemetry.tracing.score_emitter import (
    batch_score_summary_metadata,
    emit_batch_score_summary,
    emit_score_for_handle,
    emit_task_score_summary,
)


class EvaluationRecord(BaseModel):
    """
    Derived per-task evaluation result from one run.

    This is not canonical benchmark state; Langfuse datasets and their versions
    own canonical task data. Evaluation records package prediction, gold,
    scoring, metadata, and execution details for comparison, export, analysis,
    and orchestration feedback.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_config: str
    dataset_split: str
    dataset_version: str | None = None
    identity: str
    prediction: LocalizationPrediction
    gold: LocalizationGold
    score_summary: TaskScoreSummary
    repo_owner: str
    repo_name: str
    base_sha: str
    trace_id: str | None = None
    experiment_run_id: str | None = None
    evidence: LocalizationEvidence | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    materialization: RepoMaterializationResult | None = None
    token_usage: TokenUsageRecord | None = None


class EvaluationBundle(BaseModel):
    """Persistable collection of derived evaluation records for one run."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_version: str | None = None
    created_at: str | None = None
    records: list[EvaluationRecord] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    failure: LocalizationEvaluationFailure | None = None


def _safe_update_metadata(span: object | None, metadata: Mapping[str, object]) -> None:
    updater = getattr(span, "update", None)
    if callable(updater):
        try:
            updater(metadata=metadata)
        except Exception:
            pass


def _aggregate_score_summaries(
    records: Sequence[EvaluationRecord],
) -> BatchScoreSummary | None:
    if not records:
        return None
    total_count = len(records)
    composed_values = [
        record.score_summary.composed_score
        for record in records
        if record.score_summary.composed_score is not None
    ]
    component_names = sorted(
        {name for record in records for name in record.score_summary.component_scores}
    )
    components: dict[str, AggregateComponentScore] = {}
    for name in component_names:
        values = [
            record.score_summary.component_scores[name]
            for record in records
            if record.score_summary.component_scores.get(name) is not None
        ]
        numeric_values = [
            float(value) for value in values if isinstance(value, (int, float))
        ]
        available_count = len(numeric_values)
        components[name] = AggregateComponentScore(
            name=name,
            average=mean(numeric_values) if numeric_values else None,
            available_count=available_count,
            missing_count=total_count - available_count,
            total_count=total_count,
            visible_to_agent=any(
                name in record.score_summary.visible_component_scores
                for record in records
            ),
        )
    return BatchScoreSummary(
        aggregate_composed_score=mean(composed_values) if composed_values else None,
        composed_available_count=len(composed_values),
        composed_missing_count=total_count - len(composed_values),
        total_count=total_count,
        components=components,
    )


def emit_baseline_dataset_aggregates(
    span: object | None,
    records: Sequence[EvaluationRecord],
    extra_metadata: Mapping[str, object] | None = None,
) -> None:
    """
    Emit aggregate baseline metrics to the dataset/root span for observability.
    """
    aggregate_summary = _aggregate_score_summaries(records)
    if span is None or aggregate_summary is None:
        return
    emit_batch_score_summary(
        span,
        aggregate_summary,
        prefix="baseline",
        metadata_key="score_summary",
    )
    summary_metadata: dict[str, object] = {
        "selected_count": len(records),
    }
    summary_metadata.update(
        batch_score_summary_metadata(aggregate_summary, metadata_key="score_summary")
    )
    if extra_metadata:
        summary_metadata.update(dict(extra_metadata))
    _safe_update_metadata(span, summary_metadata)


def index_baselines(
    bundle: EvaluationBundle | Sequence[EvaluationRecord],
) -> dict[str, EvaluationRecord]:
    records = bundle.records if isinstance(bundle, EvaluationBundle) else list(bundle)
    return {record.identity: record for record in records}


def baseline_cache_path(
    dataset_name: str,
    dataset_version: str | None = None,
    root: Path | None = None,
) -> Path:
    """
    Derive a deterministic evaluation bundle cache path for a Langfuse dataset version.
    """
    base_root = Path(
        root
        or os.environ.get("HARNESS_BASELINE_CACHE_DIR")
        or Path.home() / ".cache" / "harness" / "baselines"
    )

    def _clean(part: str | None, default: str) -> str:
        raw = part or default
        safe = raw.replace("/", "__").replace(" ", "_")
        return safe or default

    dataset_part = _clean(dataset_name, "dataset")
    version_part = _clean(dataset_version, "latest")
    return base_root / dataset_part / f"{version_part}.json"


def persist_baseline_bundle(
    bundle: EvaluationBundle,
    *,
    dataset_name: str,
    dataset_version: str | None = None,
    cache_root: Path | None = None,
) -> Path:
    """
    Persist an evaluation bundle to the baseline cache path and return the path.
    """
    path = baseline_cache_path(
        dataset_name,
        dataset_version=dataset_version,
        root=cache_root,
    )
    save_baseline_bundle(bundle, path)
    return path


def load_persisted_baseline_bundle(
    *,
    dataset_name: str,
    dataset_version: str | None = None,
    cache_root: Path | None = None,
) -> EvaluationBundle | None:
    """
    Load a cached evaluation bundle from the baseline cache path if present.
    """
    path = baseline_cache_path(
        dataset_name,
        dataset_version=dataset_version,
        root=cache_root,
    )
    if not path.exists():
        return None
    return load_baseline_bundle(path)


def compute_baseline_for_task(
    task: LCATask,
    session_id: str | None = None,
    dataset_version: str | None = None,
    parent_trace: object | None = None,
    runner: Callable[
        [LCATask, str, object | None], LocalizationRunResult
    ]
    | None = None,
    dataset_provenance: str | None = None,
    worktree_manager: WorktreeManager | None = None,
) -> EvaluationRecord:
    with propagate_context(session_id=session_id):
        with start_observation(
            name="localization_baseline_item",
            parent=parent_trace,
            metadata={
                "dataset": task.identity.dataset_name,
                "dataset_config": task.identity.dataset_config,
                "dataset_split": task.identity.dataset_split,
                "dataset_version": dataset_version,
                "identity": task.task_id,
            },
        ) as trace:
            eval_result = evaluate_localization_batch(
                tasks=[task],
                dataset_provenance=dataset_provenance,
                worktree_manager=worktree_manager,
                parent_trace=trace,
                runner=runner,
            )
            if eval_result.failure or not eval_result.items:
                failure = eval_result.failure or LocalizationEvaluationFailure(
                    category=LocalizationFailureCategory.UNKNOWN,
                    message="baseline_evaluation_failed",
                    task_id=task.task_id,
                )
                raise LocalizationEvaluationError(
                    failure.category,
                    failure.message,
                    task_id=failure.task_id or task.task_id,  # type: ignore[arg-type]
                )
            task_result = eval_result.items[0]
            prediction_model = task_result.prediction
            score_summary = summarize_task_score(task.task_id, task_result.score_bundle)
            evidence = task_result.evidence
            materialization = task_result.materialization
            telemetry = build_localization_telemetry(
                task.identity,
                score_summary=score_summary,
                changed_files_count=len(task.gold.normalized_changed_files()),
                repo_language=task.context.repo_language,
                repo_license=task.context.repo_license,
                evidence=evidence,
                materialization_events=materialization.events
                if materialization
                else None,
            )
            emit_task_score_summary(
                trace,
                score_summary,
                prefix="baseline",
                metadata_key="score_summary",
                emit_fn=emit_score_for_handle,
            )
            return EvaluationRecord(
                dataset_name=task.identity.dataset_name,
                dataset_config=task.identity.dataset_config,
                dataset_split=task.identity.dataset_split,
                dataset_version=dataset_version,
                identity=task.task_id,
                prediction=prediction_model,
                gold=LocalizationGold(
                    changed_files=task.gold.normalized_changed_files()
                ),
                score_summary=score_summary,
                repo_owner=task.identity.repo_owner,
                repo_name=task.identity.repo_name,
                base_sha=task.identity.base_sha,
                trace_id=getattr(trace, "id", None),
                experiment_run_id=None,
                evidence=evidence,
                token_usage=task_result.token_usage,
                metadata={
                    "telemetry": telemetry.model_dump(exclude_none=True),
                    "score_summary": score_summary.model_dump(mode="json"),
                    "run_kind": "localization_baseline",
                    "dataset_provenance": getattr(parent_trace, "metadata", {}).get(
                        "dataset_provenance"
                    )
                    if parent_trace
                    else None,
                },
                materialization=materialization,
            )


def resolve_baseline(
    baselines: EvaluationBundle | Sequence[EvaluationRecord], task: LCATask
) -> EvaluationRecord | None:
    indexed = index_baselines(baselines)
    return indexed.get(task.task_id)


def require_baseline(
    baselines: EvaluationBundle | Sequence[EvaluationRecord], task: LCATask
) -> EvaluationRecord:
    resolved = resolve_baseline(baselines, task)
    if resolved is None:
        raise ValueError(
            f"No baseline record found for localization task_id={task.task_id}"
        )
    return resolved


def make_baseline_bundle(
    records: list[EvaluationRecord] | Sequence[Mapping[str, object]],
    dataset_name: str,
    dataset_version: str | None = None,
    metadata: Mapping[str, object] | None = None,
    failure: LocalizationEvaluationFailure | None = None,
) -> EvaluationBundle:
    record_models = [
        record
        if isinstance(record, EvaluationRecord)
        else EvaluationRecord.model_validate(record)
        for record in records
    ]
    return EvaluationBundle(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        records=record_models,
        metadata=dict(metadata) if metadata else {},
        failure=failure,
    )


def save_baseline_bundle(bundle: EvaluationBundle, path: Path) -> None:
    serializable = bundle.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def load_baseline_bundle(path: Path) -> EvaluationBundle:
    data: object = json.loads(path.read_text())
    try:
        return EvaluationBundle.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid evaluation bundle: {exc}") from exc
