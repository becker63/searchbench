from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, cast

from pydantic import BaseModel, Field

from .datasets import DatasetItem, normalize_dataset_item
from .langfuse import record_score, start_span
from ..scorer import score
from ..utils.repo_targets import resolve_repo_target


def _stable_repo_symbol(repo: str, symbol: str) -> str:
    return hashlib.sha256(f"{repo}::{symbol}".encode()).hexdigest()


class BaselineSnapshot(BaseModel):
    dataset_name: str | None = None
    dataset_version: str | None = None
    item_id: str | None = None
    repo: str
    symbol: str
    jc_result: dict[str, object] = Field(default_factory=dict)
    jc_metrics: dict[str, float] = Field(default_factory=dict)
    experiment_run_id: str | None = None
    trace_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class BaselineBundle(BaseModel):
    dataset_name: str | None = None
    dataset_version: str | None = None
    baseline_run_id: str | None = None
    baseline_run_name: str | None = None
    dataset_run_id: str | None = None
    created_at: str | None = None
    items: list[BaselineSnapshot] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


def _item_identity(item: DatasetItem) -> str:
    normalized = normalize_dataset_item(item)
    if normalized.id:
        return normalized.id
    return hashlib.sha256(f"{normalized.repo}::{normalized.symbol}".encode()).hexdigest()


def baseline_key(item: DatasetItem) -> str:
    return _item_identity(item)


def _snapshot_key(snapshot: BaselineSnapshot) -> str:
    if snapshot.item_id:
        return snapshot.item_id
    if snapshot.repo and snapshot.symbol:
        return _stable_repo_symbol(snapshot.repo, snapshot.symbol)
    return ""


def baseline_metrics_from_jc(jc_result: Mapping[str, object]) -> dict[str, float]:
    combined = {"iterative_context": {"observations": []}, "jcodemunch": jc_result}
    metrics = score(combined)
    numeric_metrics: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, bool)):
            numeric_metrics[k] = float(v)
    return numeric_metrics


def index_baselines(bundle: BaselineBundle | Sequence[BaselineSnapshot]) -> dict[str, BaselineSnapshot]:
    items = bundle.items if isinstance(bundle, BaselineBundle) else list(bundle)
    indexed: dict[str, BaselineSnapshot] = {}
    for snap in items:
        key = _snapshot_key(snap)
        if key:
            indexed[key] = snap
    return indexed


def compute_baseline_for_item(
    item: DatasetItem,
    dataset_name: str | None = None,
    dataset_version: str | None = None,
    parent_trace=None,
) -> BaselineSnapshot:
    from ..runner import run_jc_iteration  # local import to avoid circular dependency

    normalized = normalize_dataset_item(item)
    repo_ref = normalized.repo
    resolved_repo = resolve_repo_target(repo_ref)
    trace = start_span(
        parent_trace,
        "jc_baseline_item",
        metadata={
            "dataset": dataset_name,
            "dataset_version": dataset_version,
            "item_id": normalized.id,
            "repo": normalized.repo,
            "symbol": normalized.symbol,
            "run_kind": "jc_baseline",
            "repo_ref": repo_ref,
            "resolved_repo": resolved_repo,
        },
    )
    jc_result = run_jc_iteration({"symbol": normalized.symbol, "repo": resolved_repo}, parent_trace=trace)
    metrics = baseline_metrics_from_jc(cast(Mapping[str, object], jc_result))
    for name, value in metrics.items():
        record_score(trace, f"baseline.{name}", value)
    if trace and hasattr(trace, "end"):
        try:
            trace.end(metadata={"jc_nodes": metrics.get("jc_nodes", 0)})
        except Exception:
            pass
    return BaselineSnapshot(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        item_id=normalized.id,
        repo=normalized.repo,
        symbol=normalized.symbol,
        jc_result=cast(dict[str, object], jc_result),
        jc_metrics=metrics,
        trace_id=getattr(trace, "id", None),
        experiment_run_id=getattr(parent_trace, "id", None),
        metadata={"run_kind": "jc_baseline"},
    )


def resolve_baseline(baselines: BaselineBundle | Sequence[BaselineSnapshot], item: DatasetItem) -> BaselineSnapshot | None:
    identity = baseline_key(item)
    snapshots = baselines.items if isinstance(baselines, BaselineBundle) else list(baselines)
    for snap in snapshots:
        if _snapshot_key(snap) == identity:
            return snap
    for snap in snapshots:
        if snap.repo == item.repo and snap.symbol == item.symbol:
            return snap
    return None


def require_baseline(baselines: BaselineBundle | Sequence[BaselineSnapshot], item: DatasetItem) -> BaselineSnapshot:
    resolved = resolve_baseline(baselines, item)
    if resolved is None:
        raise ValueError(f"No baseline snapshot found for repo={item.repo} symbol={item.symbol}")
    return resolved


def make_baseline_bundle(
    snapshots: list[BaselineSnapshot] | Sequence[Mapping[str, object]],
    dataset_name: str | None = None,
    dataset_version: str | None = None,
    baseline_run_id: str | None = None,
    baseline_run_name: str | None = None,
    dataset_run_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BaselineBundle:
    snapshot_models = [
        snap if isinstance(snap, BaselineSnapshot) else BaselineSnapshot.model_validate(snap) for snap in snapshots
    ]
    return BaselineBundle(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        baseline_run_id=baseline_run_id,
        baseline_run_name=baseline_run_name,
        dataset_run_id=dataset_run_id or baseline_run_id,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        items=snapshot_models,
        metadata=dict(metadata) if metadata else {},
    )


def save_baseline_bundle(bundle: BaselineBundle, path: Path) -> None:
    serializable = bundle.model_dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def load_baseline_bundle(path: Path) -> BaselineBundle:
    data: object = json.loads(path.read_text())
    return BaselineBundle.model_validate(data)
