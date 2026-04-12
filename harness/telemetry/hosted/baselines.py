from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.localization.runtime.evaluate import (
    LocalizationEvaluationFailure,
    evaluate_localization_batch,
)
from harness.localization.materialization.materialize import RepoMaterializer, RepoMaterializationResult
from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory
from harness.localization.models import (
    LCATask,
    LocalizationEvidence,
    LocalizationGold,
    LocalizationMetrics,
    LocalizationPrediction,
    normalize_lca_task,
)
from harness.localization.token_usage import TokenUsageRecord
from harness.localization.telemetry import build_localization_telemetry
from harness.telemetry.tracing import propagate_context, start_observation
from harness.telemetry.tracing.score_emitter import emit_score_for_handle


class BaselineSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_config: str
    dataset_split: str
    dataset_version: str | None = None
    identity: str
    prediction: LocalizationPrediction
    gold: LocalizationGold
    metrics: LocalizationMetrics
    repo_owner: str
    repo_name: str
    base_sha: str
    trace_id: str | None = None
    experiment_run_id: str | None = None
    evidence: LocalizationEvidence | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    materialization: RepoMaterializationResult | None = None
    token_usage: TokenUsageRecord | None = None


class BaselineBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_version: str | None = None
    created_at: str | None = None
    items: list[BaselineSnapshot] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    failure: LocalizationEvaluationFailure | None = None


def _safe_update_metadata(span: object | None, metadata: Mapping[str, object]) -> None:
    updater = getattr(span, "update", None)
    if callable(updater):
        try:
            updater(metadata=metadata)
        except Exception:
            pass


def _aggregate_metrics(snapshots: Sequence[BaselineSnapshot]) -> LocalizationMetrics | None:
    metrics = [snap.metrics for snap in snapshots if isinstance(snap.metrics, LocalizationMetrics)]
    if not metrics:
        return None
    return LocalizationMetrics(
        precision=mean(m.precision for m in metrics),
        recall=mean(m.recall for m in metrics),
        f1=mean(m.f1 for m in metrics),
        hit=mean(m.hit for m in metrics),
        score=mean(m.score for m in metrics),
    )


def emit_baseline_dataset_aggregates(
    span: object | None,
    snapshots: Sequence[BaselineSnapshot],
    extra_metadata: Mapping[str, object] | None = None,
) -> None:
    """
    Emit aggregate baseline metrics to the dataset/root span for observability.
    """
    aggregates = _aggregate_metrics(snapshots)
    if span is None or aggregates is None:
        return
    metrics_payload = {
        "precision": aggregates.precision,
        "recall": aggregates.recall,
        "f1": aggregates.f1,
        "hit": aggregates.hit,
        "score": aggregates.score,
    }
    for metric_name, metric_value in metrics_payload.items():
        emit_score_for_handle(
            span,
            name=f"baseline.avg_{metric_name}",
            value=float(metric_value),
            data_type="NUMERIC",
            score_id=f"{getattr(span, 'id', None)}-baseline.avg_{metric_name}" if getattr(span, "id", None) else None,
        )
    summary_metadata: dict[str, object] = {
        "selected_count": len(snapshots),
        "aggregate_metrics": metrics_payload,
    }
    if extra_metadata:
        summary_metadata.update(dict(extra_metadata))
    _safe_update_metadata(span, summary_metadata)


def index_baselines(bundle: BaselineBundle | Sequence[BaselineSnapshot]) -> dict[str, BaselineSnapshot]:
    items = bundle.items if isinstance(bundle, BaselineBundle) else list(bundle)
    return {snap.identity: snap for snap in items}


def baseline_cache_path(
    dataset_name: str,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    dataset_version: str | None = None,
    root: Path | None = None,
) -> Path:
    """
    Derive a deterministic baseline bundle cache path based on dataset identity.
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
    config_part = _clean(dataset_config, "config")
    split_part = _clean(dataset_split, "split")
    version_part = _clean(dataset_version, "latest")
    return base_root / dataset_part / config_part / split_part / f"{version_part}.json"


def persist_baseline_bundle(
    bundle: BaselineBundle,
    *,
    dataset_name: str,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    dataset_version: str | None = None,
    cache_root: Path | None = None,
) -> Path:
    """
    Persist a baseline bundle to the canonical cache path and return the path.
    """
    path = baseline_cache_path(
        dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        dataset_version=dataset_version,
        root=cache_root,
    )
    save_baseline_bundle(bundle, path)
    return path


def load_persisted_baseline_bundle(
    *,
    dataset_name: str,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    dataset_version: str | None = None,
    cache_root: Path | None = None,
) -> BaselineBundle | None:
    """
    Load a cached baseline bundle from the canonical path if present.
    """
    path = baseline_cache_path(
        dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
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
    runner: Callable[[LCATask, str, object | None], tuple[list[str], Mapping[str, object] | None]] | None = None,
    dataset_source: str | None = None,
    materializer: RepoMaterializer | None = None,
) -> BaselineSnapshot:
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
                dataset_source=dataset_source,
                materializer=materializer,
                parent_trace=trace,
                runner=runner,
            )
            if eval_result.failure or not eval_result.items:
                failure = eval_result.failure or LocalizationEvaluationFailure(
                    category=LocalizationFailureCategory.UNKNOWN, message="baseline_evaluation_failed", task_id=task.task_id
                )
                raise LocalizationEvaluationError(
                    failure.category, failure.message, task_id=failure.task_id or task.task_id  # type: ignore[arg-type]
                )
            task_result = eval_result.items[0]
            prediction_model = LocalizationPrediction(predicted_files=task_result.prediction)
            metrics = task_result.metrics
            evidence = task_result.evidence
            materialization = task_result.materialization
            telemetry = build_localization_telemetry(
                task.identity,
                metrics=metrics,
                changed_files_count=len(task.gold.normalized_changed_files()),
                repo_language=task.context.repo_language,
                repo_license=task.context.repo_license,
                evidence=evidence,
                materialization_events=materialization.events if materialization else None,
            )
            for name, value in metrics.model_dump().items():
                if value is None:
                    continue
                emit_score_for_handle(
                    trace,
                    name=f"baseline.{name}",
                    value=float(value),
                    data_type="NUMERIC",
                    score_id=f"{trace.id}-baseline.{name}" if getattr(trace, "id", None) else None,
                )
            return BaselineSnapshot(
                dataset_name=task.identity.dataset_name,
                dataset_config=task.identity.dataset_config,
                dataset_split=task.identity.dataset_split,
                dataset_version=dataset_version,
                identity=task.task_id,
                prediction=prediction_model,
                gold=LocalizationGold(changed_files=task.gold.normalized_changed_files()),
                metrics=metrics,
                repo_owner=task.identity.repo_owner,
                repo_name=task.identity.repo_name,
                base_sha=task.identity.base_sha,
                trace_id=getattr(trace, "id", None),
                experiment_run_id=None,
                evidence=evidence,
                token_usage=task_result.token_usage,
                metadata={
                    "telemetry": telemetry.model_dump(exclude_none=True),
                    "run_kind": "localization_baseline",
                    "dataset_source": getattr(parent_trace, "metadata", {}).get("dataset_source") if parent_trace else None,
                },
                materialization=materialization,
            )


def resolve_baseline(baselines: BaselineBundle | Sequence[BaselineSnapshot], task: LCATask) -> BaselineSnapshot | None:
    indexed = index_baselines(baselines)
    return indexed.get(task.task_id)


def require_baseline(baselines: BaselineBundle | Sequence[BaselineSnapshot], task: LCATask) -> BaselineSnapshot:
    resolved = resolve_baseline(baselines, task)
    if resolved is None:
        raise ValueError(f"No baseline snapshot found for localization task_id={task.task_id}")
    return resolved


def make_baseline_bundle(
    snapshots: list[BaselineSnapshot] | Sequence[Mapping[str, object]],
    dataset_name: str,
    dataset_version: str | None = None,
    metadata: Mapping[str, object] | None = None,
    failure: LocalizationEvaluationFailure | None = None,
) -> BaselineBundle:
    snapshot_models = [
        snap if isinstance(snap, BaselineSnapshot) else BaselineSnapshot.model_validate(snap) for snap in snapshots
    ]
    return BaselineBundle(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        items=snapshot_models,
        metadata=dict(metadata) if metadata else {},
        failure=failure,
    )


def save_baseline_bundle(bundle: BaselineBundle, path: Path) -> None:
    serializable = bundle.model_dump(exclude_none=True, mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def load_baseline_bundle(path: Path) -> BaselineBundle:
    data: object = json.loads(path.read_text())
    try:
        return BaselineBundle.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid baseline bundle: {exc}") from exc


def normalize_baseline_task(data: Mapping[str, object]) -> LCATask:
    """
    Helper to coerce legacy mappings into LCATask for baseline compatibility.
    """
    return normalize_lca_task(data)
