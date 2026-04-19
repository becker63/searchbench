from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .store import GraphStore


@dataclass
class HopDistanceResult:
    hop_by_prediction: dict[str, int | None]
    min_hop: int | None


def compute_hop_distance_summary(
    store: GraphStore, anchor_nodes: Iterable[str], prediction_nodes: dict[str, list[str]]
) -> HopDistanceResult:
    """
    Compute minimal hop distances from any anchor node to each prediction file node.
    Returns per-prediction hops and the overall minimum hop observed.
    """
    anchors = list(anchor_nodes)
    per_pred: dict[str, int | None] = {}
    best: int | None = None
    for pred_path, nodes in prediction_nodes.items():
        hop = store.shortest_hop(anchors, nodes[0]) if nodes else None
        per_pred[pred_path] = hop
        if hop is not None and (best is None or hop < best):
            best = hop
    return HopDistanceResult(hop_by_prediction=per_pred, min_hop=best)
