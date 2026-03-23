from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable


def _safe_get_nodes(bucket: object) -> list[Any]:
    if not isinstance(bucket, Mapping):
        return []
    nodes = bucket.get("nodes")
    return nodes if isinstance(nodes, list) else []


def _normalize_nodes(result: Mapping[str, object], key: str) -> list[Any]:
    bucket = result.get(key)
    if not isinstance(bucket, Mapping):
        return []
    if "nodes" in bucket:
        return _safe_get_nodes(bucket)
    if "graph" in bucket and isinstance(bucket["graph"], Mapping):
        return _safe_get_nodes(bucket["graph"])
    if "bundle" in bucket and isinstance(bucket["bundle"], Mapping):
        symbols = bucket["bundle"].get("symbols")
        return symbols if isinstance(symbols, list) else []
    return []


def _node_identifier(node: Any) -> str | None:
    if isinstance(node, Mapping):
        for key in ("id", "symbol_id", "symbol", "name"):
            val = node.get(key)
            if isinstance(val, (str, int)):
                return str(val)
    return None


def _unique_count(nodes: Iterable[Any]) -> int:
    seen: set[str] = set()
    unique = 0
    for idx, node in enumerate(nodes):
        ident = _node_identifier(node)
        if ident is None:
            ident = f"_anon_{idx}"
        if ident in seen:
            continue
        seen.add(ident)
        unique += 1
    return unique


def score(result: Mapping[str, object]) -> Dict[str, object]:
    """
    Deterministic scoring that rewards unique, efficient exploration.
    """
    ic_nodes_list = _normalize_nodes(result, "iterative_context")
    jc_nodes_list = _normalize_nodes(result, "jcodemunch")

    ic_total = len(ic_nodes_list)
    jc_total = len(jc_nodes_list)

    ic_unique = _unique_count(ic_nodes_list)
    jc_unique = _unique_count(jc_nodes_list)

    # Advantage of iterative-context over jcodemunch in unique discoveries.
    coverage_delta = ic_unique - jc_unique

    # Efficiency: unique nodes per explored node (higher is better).
    efficiency_ic = ic_unique / max(ic_total, 1)

    # Redundancy: proportion of duplicate exploration (higher is worse).
    redundancy_ic = 1.0 - efficiency_ic if ic_total else 0.0

    score_val = (
        coverage_delta * 1.0
        + efficiency_ic * 0.5
        - redundancy_ic * 0.25
    )

    return {
        "ic_nodes": ic_total,
        "jc_nodes": jc_total,
        "unique_ic_nodes": ic_unique,
        "unique_jc_nodes": jc_unique,
        "coverage_delta": coverage_delta,
        "efficiency_ic": efficiency_ic,
        "redundancy_ic": redundancy_ic,
        "score": float(score_val),
    }
