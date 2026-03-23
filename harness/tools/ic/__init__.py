from __future__ import annotations

from typing import Any, Dict, Optional

from iterative_context import exploration


def resolve(symbol: str) -> Optional[Dict[str, Any]]:
    exploration.ensure_graph_loaded()
    node: Any = exploration.resolve(symbol)
    if node is None:
        return None

    graph = exploration.get_active_graph()
    extras: Dict[str, Any] = {}
    node_id = getattr(node, "id", None)
    if graph is not None and node_id in getattr(graph, "nodes", {}):
        raw = graph.nodes[node_id]
        extras = {k: v for k, v in raw.items() if k != "data"}

    return {
        "id": node_id,
        "type": getattr(node, "type", None),
        "data": getattr(node, "data", None),
        "extras": extras,
    }


def expand(node_id: str, depth: int, score_fn):
    exploration.ensure_graph_loaded()
    return exploration.expand(node_id=node_id, depth=depth, score_fn=score_fn)


def resolve_and_expand(symbol: str, depth: int, score_fn):
    node = resolve(symbol)
    if node is None:
        return {"nodes": [], "edges": [], "metadata": {}}
    return expand(node["id"], depth, score_fn)
