from __future__ import annotations

from typing import Mapping

from pydantic import BaseModel, Field

from harness.localization.executor import run_localization_task
from harness.localization.models import LCATask, LocalizationMetrics, normalize_lca_task
from harness.localization.materializer import RepoMaterializationResult, RepoMaterializer
from harness.observability.baselines import BaselineBundle, BaselineSnapshot, compute_baseline_for_task, make_baseline_bundle
from harness.observability.datasets import (
    fetch_localization_dataset_from_source,
    normalize_localization_task,
)
from harness.observability.langfuse import flush_langfuse, propagate_context
from harness.observability.session_policy import resolve_session_id

from .requests import HostedLocalizationBaselineRequest, HostedLocalizationRunRequest


class LocalizationRunItemResult(BaseModel):
    task: Mapping[str, object]
    metrics: LocalizationMetrics
    prediction: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    materialization: RepoMaterializationResult | None = None


class LocalizationRunResult(BaseModel):
    dataset_name: str
    dataset_version: str | None = None
    items: list[LocalizationRunItemResult] = Field(default_factory=list)


def _resolve_items(req: HostedLocalizationRunRequest) -> list[LCATask]:
    return fetch_localization_dataset_from_source(
        source=req.dataset_source,
        name=req.dataset,
        version=req.version,
        dataset_config=req.dataset_config,
        dataset_split=req.dataset_split,
    )


def run_hosted_localization_baseline(req: HostedLocalizationBaselineRequest, materializer: RepoMaterializer | None = None) -> BaselineBundle:
    tasks = fetch_localization_dataset_from_source(
        source=req.dataset_source,
        name=req.dataset,
        version=req.version,
        dataset_config=req.dataset_config,
        dataset_split=req.dataset_split,
    )
    if not tasks:
        raise RuntimeError("No localization tasks available for baseline run")
    snapshots: list[BaselineSnapshot] = []
    with propagate_context(
        trace_name="localization_baseline_dataset",
        session_id=resolve_session_id(req.session, run_id=req.dataset),
        metadata={
            "dataset": req.dataset,
            "dataset_version": req.version,
            "dataset_config": req.dataset_config,
            "dataset_split": req.dataset_split,
            "run_kind": "localization_baseline",
            "dataset_source": req.dataset_source.value,
        },
    ) as trace:
        for task in tasks:
            snapshots.append(
                compute_baseline_for_task(
                    task,
                    dataset_version=req.version,
                    parent_trace=trace,
                    dataset_source=req.dataset_source.value,
                    materializer=materializer,
                )
            )
    flush_langfuse()
    return make_baseline_bundle(
        snapshots,
        dataset_name=req.dataset,
        dataset_version=req.version,
        metadata={"run_kind": "localization_baseline"},
    )


def run_hosted_localization_experiment(req: HostedLocalizationRunRequest, materializer: RepoMaterializer | None = None) -> LocalizationRunResult:
    tasks = _resolve_items(req)
    if not tasks:
        raise RuntimeError("No localization tasks available for experiment run")
    results: list[LocalizationRunItemResult] = []
    with propagate_context(
        trace_name="localization_experiment",
        session_id=resolve_session_id(req.session, run_id=req.dataset),
        metadata={
            "dataset": req.dataset,
            "dataset_version": req.version,
            "dataset_config": req.dataset_config,
            "dataset_split": req.dataset_split,
            "run_kind": "localization_experiment",
            "dataset_source": req.dataset_source.value,
        },
    ) as trace:
        for task in tasks:
            prediction, metrics, _evidence, mat = run_localization_task(
                task, dataset_source=req.dataset_source.value, materializer=materializer, parent_trace=trace
            )
            results.append(
                LocalizationRunItemResult(
                    task=normalize_lca_task(task).model_dump(),
                    metrics=metrics,
                    prediction=prediction.predicted_files,
                    trace_id=getattr(trace, "id", None),
                    materialization=mat,
                )
            )
    flush_langfuse()
    return LocalizationRunResult(dataset_name=req.dataset, dataset_version=req.version, items=results)
