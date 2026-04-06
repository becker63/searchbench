from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.localization.evaluation_backend import (
    LocalizationEvaluationFailure,
    LocalizationEvaluationRequest,
    evaluate_localization_batch,
)
from harness.localization.materializer import RepoMaterializer, RepoMaterializationResult
from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory
from harness.localization.models import (
    LCATask,
    LocalizationEvidence,
    LocalizationGold,
    LocalizationMetrics,
    LocalizationPrediction,
    normalize_lca_task,
)
from harness.localization.telemetry import build_localization_telemetry
from harness.observability.langfuse import start_observation
from harness.observability.score_emitter import emit_score_for_handle


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


class BaselineBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_version: str | None = None
    created_at: str | None = None
    items: list[BaselineSnapshot] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    failure: LocalizationEvaluationFailure | None = None


def baseline_key(task: LCATask) -> str:
    return task.task_id


def index_baselines(bundle: BaselineBundle | Sequence[BaselineSnapshot]) -> dict[str, BaselineSnapshot]:
    items = bundle.items if isinstance(bundle, BaselineBundle) else list(bundle)
    return {snap.identity: snap for snap in items}


def compute_baseline_for_task(
    task: LCATask,
    dataset_version: str | None = None,
    parent_trace: object | None = None,
    runner: Callable[[LCATask, str, object | None], tuple[list[str], Mapping[str, object] | None]] | None = None,
    dataset_source: str | None = None,
    materializer: RepoMaterializer | None = None,
) -> BaselineSnapshot:
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
            LocalizationEvaluationRequest(
                tasks=[task],
                dataset_source=dataset_source,
                materializer=materializer,
                parent_trace=trace,
                runner=runner,
            )
        )
        if eval_result.failure or not eval_result.items:
            failure = eval_result.failure or LocalizationEvaluationFailure(
                category=LocalizationFailureCategory.UNKNOWN, message="baseline_evaluation_failed", task_id=task.task_id
            )
            raise LocalizationEvaluationError(
                failure.category, failure.message, task_id=failure.task_id or task.task_id  # type: ignore[arg-type]
            )
        task_result = eval_result.items[0]
        prediction = LocalizationPrediction(predicted_files=task_result.prediction)
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
        dataset_run_id = getattr(parent_trace, "id", None)
        for name, value in metrics.model_dump().items():
            if value is None:
                continue
            emit_score_for_handle(
                trace,
                name=f"baseline.{name}",
                value=float(value),
                data_type="NUMERIC",
                dataset_run_id=dataset_run_id,
                score_id=f"{trace.id}-baseline.{name}" if getattr(trace, "id", None) else None,
            )
        return BaselineSnapshot(
            dataset_name=task.identity.dataset_name,
            dataset_config=task.identity.dataset_config,
            dataset_split=task.identity.dataset_split,
            dataset_version=dataset_version,
            identity=task.task_id,
            prediction=LocalizationPrediction(predicted_files=prediction.predicted_files),
            gold=LocalizationGold(changed_files=task.gold.normalized_changed_files()),
            metrics=metrics,
            repo_owner=task.identity.repo_owner,
            repo_name=task.identity.repo_name,
            base_sha=task.identity.base_sha,
            trace_id=getattr(trace, "id", None),
            experiment_run_id=dataset_run_id,
            evidence=evidence,
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
    serializable = bundle.model_dump(exclude_none=True)
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
