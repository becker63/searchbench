from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, TypedDict, cast

from .datasets import DatasetItem, normalize_dataset_item
from .langfuse import record_score, start_span
from ..scorer import score


class BaselineSnapshot(TypedDict, total=False):
    dataset_name: str | None
    dataset_version: str | None
    item_id: str | None
    repo: str
    symbol: str
    jc_result: dict[str, object]
    jc_metrics: dict[str, float]
    experiment_run_id: str | None
    trace_id: str | None
    metadata: dict[str, object]


class BaselineBundle(TypedDict, total=False):
    dataset_name: str | None
    dataset_version: str | None
    baseline_run_id: str | None
    baseline_run_name: str | None
    dataset_run_id: str | None
    created_at: str | None
    items: list[BaselineSnapshot]
    metadata: dict[str, object]


def _stable_repo_symbol(repo: str, symbol: str) -> str:
    return hashlib.sha256(f"{repo}::{symbol}".encode()).hexdigest()


def _item_identity(item: DatasetItem) -> str:
    normalized = normalize_dataset_item(item)
    if normalized.get("id"):
        return cast(str, normalized.get("id"))
    h = hashlib.sha256(f"{normalized.get('repo')}::{normalized.get('symbol')}".encode()).hexdigest()
    return h


def baseline_key(item: DatasetItem) -> str:
    return _item_identity(item)


def _snapshot_key(snapshot: BaselineSnapshot) -> str:
    snap_id = snapshot.get("item_id")
    if isinstance(snap_id, str):
        return snap_id
    repo = snapshot.get("repo")
    symbol = snapshot.get("symbol")
    if isinstance(repo, str) and isinstance(symbol, str):
        return _stable_repo_symbol(repo, symbol)
    return ""


def baseline_metrics_from_jc(jc_result: Mapping[str, object]) -> dict[str, float]:
    combined = {"iterative_context": {"observations": []}, "jcodemunch": jc_result}
    metrics = score(combined)
    numeric_metrics: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, bool)):
            numeric_metrics[k] = float(v)
    return numeric_metrics


def index_baselines(bundle: BaselineBundle | list[BaselineSnapshot]) -> dict[str, BaselineSnapshot]:
    items = bundle["items"] if isinstance(bundle, Mapping) and "items" in bundle else bundle  # type: ignore[index]
    indexed: dict[str, BaselineSnapshot] = {}
    for snap in items:
        if isinstance(snap, Mapping):
            snapshot = cast(BaselineSnapshot, snap)
            key = _snapshot_key(snapshot)
            if key:
                indexed[key] = snapshot
    return indexed


def compute_baseline_for_item(
    item: DatasetItem,
    dataset_name: str | None = None,
    dataset_version: str | None = None,
    parent_trace=None,
) -> BaselineSnapshot:
    from ..runner import run_jc_iteration  # local import to avoid circular dependency
    normalized = normalize_dataset_item(item)
    trace = start_span(
        parent_trace,
        "jc_baseline_item",
        metadata={
            "dataset": dataset_name,
            "dataset_version": dataset_version,
            "item_id": normalized.get("id"),
            "repo": normalized.get("repo"),
            "symbol": normalized.get("symbol"),
            "run_kind": "jc_baseline",
        },
    )
    jc_result = run_jc_iteration({"symbol": normalized["symbol"], "repo": normalized["repo"]}, parent_trace=trace)
    metrics = baseline_metrics_from_jc(jc_result)
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
        item_id=normalized.get("id"),
        repo=normalized["repo"],
        symbol=normalized["symbol"],
        jc_result=cast(dict[str, object], jc_result),
        jc_metrics=metrics,
        trace_id=getattr(trace, "id", None),
        experiment_run_id=getattr(parent_trace, "id", None),
        metadata={"run_kind": "jc_baseline"},
    )


def resolve_baseline(baselines: BaselineBundle | list[BaselineSnapshot], item: DatasetItem) -> BaselineSnapshot | None:
    identity = baseline_key(item)
    snapshots_raw = baselines["items"] if isinstance(baselines, Mapping) and "items" in baselines else baselines  # type: ignore[index]
    snapshots: list[BaselineSnapshot] = []
    for snap in snapshots_raw:
        if isinstance(snap, Mapping):
            snapshots.append(cast(BaselineSnapshot, snap))
    for snap in snapshots:
        if _snapshot_key(snap) == identity:
            return snap
    for snap in snapshots:
        if snap.get("repo") == item.get("repo") and snap.get("symbol") == item.get("symbol"):
            return snap
    return None


def require_baseline(baselines: BaselineBundle | list[BaselineSnapshot], item: DatasetItem) -> BaselineSnapshot:
    resolved = resolve_baseline(baselines, item)
    if resolved is None:
        raise ValueError(f"No baseline snapshot found for repo={item.get('repo')} symbol={item.get('symbol')}")
    return resolved


def make_baseline_bundle(
    snapshots: list[BaselineSnapshot],
    dataset_name: str | None = None,
    dataset_version: str | None = None,
    baseline_run_id: str | None = None,
    baseline_run_name: str | None = None,
    dataset_run_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BaselineBundle:
    return BaselineBundle(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        baseline_run_id=baseline_run_id,
        baseline_run_name=baseline_run_name,
        dataset_run_id=dataset_run_id or baseline_run_id,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        items=snapshots,
        metadata=dict(metadata) if metadata else {},
    )


def save_baseline_bundle(bundle: BaselineBundle, path: Path) -> None:
    serializable = json.loads(json.dumps(bundle, default=str))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def load_baseline_bundle(path: Path) -> BaselineBundle:
    data: object = json.loads(path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError("Invalid baseline bundle format")
    items_raw = data.get("items", [])
    snapshots: list[BaselineSnapshot] = []
    if isinstance(items_raw, list):
        for snap in items_raw:
            if isinstance(snap, Mapping):
                repo = snap.get("repo")
                symbol = snap.get("symbol")
                if not isinstance(repo, str) or not isinstance(symbol, str):
                    continue
                jc_result_obj = snap.get("jc_result", {})
                jc_metrics_obj = snap.get("jc_metrics", {})
                try:
                    snapshots.append(
                        BaselineSnapshot(
                            dataset_name=cast(str | None, snap.get("dataset_name") or data.get("dataset_name")),
                            dataset_version=cast(str | None, snap.get("dataset_version") or data.get("dataset_version")),
                            item_id=cast(str | None, snap.get("item_id")),
                            repo=repo,
                            symbol=symbol,
                            jc_result=jc_result_obj if isinstance(jc_result_obj, dict) else {},
                            jc_metrics=jc_metrics_obj if isinstance(jc_metrics_obj, dict) else {},
                            experiment_run_id=cast(str | None, snap.get("experiment_run_id")),
                            trace_id=cast(str | None, snap.get("trace_id")),
                            metadata=cast(dict[str, object], snap.get("metadata", {})) if isinstance(snap.get("metadata", {}), dict) else {},
                        )
                    )
                except Exception:
                    continue
    return make_baseline_bundle(
        snapshots,
        dataset_name=cast(str | None, data.get("dataset_name")),
        dataset_version=cast(str | None, data.get("dataset_version")),
        baseline_run_id=cast(str | None, data.get("baseline_run_id")),
        baseline_run_name=cast(str | None, data.get("baseline_run_name")),
        dataset_run_id=cast(str | None, data.get("dataset_run_id")),
        metadata=cast(Mapping[str, object] | None, data.get("metadata", {})) if isinstance(data.get("metadata", {}), Mapping) else None,
    )
