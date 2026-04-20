from __future__ import annotations

from typing import Any, cast

from iterative_context.graph_models import Graph, GraphEdge, GraphNode


def _unwrap_node(data: Any) -> tuple[GraphNode, dict[str, Any]]:
    if isinstance(data, dict) and "data" in data:
        typed_data = cast(dict[str, Any], data)
        node = cast(GraphNode, typed_data["data"])
        extras = {k: v for k, v in typed_data.items() if k != "data"}
        return node, extras
    if isinstance(data, dict):
        typed_data = cast(dict[str, Any], data)
        node = cast(GraphNode, typed_data)
        return node, typed_data
    return cast(GraphNode, data), {}


def serialize_node(node: GraphNode, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a GraphNode into a JSON-serializable dict."""
    extras = extras or {}
    symbol_value = extras.get("symbol")
    file_value = extras.get("file")

    payload: dict[str, Any] = {
        "id": node.id,
        "symbol": symbol_value if isinstance(symbol_value, str) else node.id,
        "kind": node.kind,
        "state": node.state,
    }
    if isinstance(file_value, str):
        payload["file"] = file_value
    tokens = getattr(node, "tokens", None)
    if tokens is not None:
        payload["tokens"] = tokens
    return payload


def serialize_graph(graph: Graph, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialize a NetworkX graph into deterministic JSON structures."""
    nodes: list[dict[str, Any]] = []
    for _, data in graph.nodes(data=True):
        node_obj, extras = _unwrap_node(data)
        nodes.append(serialize_node(node_obj, extras))

    edges: list[dict[str, Any]] = []
    for source, target, raw_data in graph.edges(data=True):
        edge_kind_value: str | None = None
        typed_edge: dict[str, Any] = raw_data
        embedded = typed_edge.get("data")
        if isinstance(embedded, GraphEdge):
            edge_kind_value = embedded.kind
        else:
            raw_kind = typed_edge.get("kind")
            edge_kind_value = raw_kind if isinstance(raw_kind, str) else None
        edges.append(
            {
                "source": cast(str, source),
                "target": cast(str, target),
                "type": edge_kind_value if isinstance(edge_kind_value, str) else "references",
            }
        )

    nodes.sort(key=lambda n: n["id"])
    edges.sort(key=lambda e: (e["source"], e["target"], e["type"]))

    return {"nodes": nodes, "edges": edges, "metadata": metadata or {}}


__all__ = ["serialize_node", "serialize_graph"]
