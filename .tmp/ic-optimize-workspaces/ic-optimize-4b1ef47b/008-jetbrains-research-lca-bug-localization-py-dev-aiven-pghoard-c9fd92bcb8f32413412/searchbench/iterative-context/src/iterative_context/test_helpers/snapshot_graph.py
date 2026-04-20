# pyright: reportMissingTypeStubs=false
import json
from types import SimpleNamespace
from typing import Any, cast

from pytest_snapshot.plugin import Snapshot

from iterative_context.graph_models import (
    AddEdgesEvent,
    AddNodesEvent,
    Graph,
    GraphEdge,
    GraphEvent,
    GraphNode,
    UpdateNodeEvent,
)


def _remove_none(d: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of *d* without keys whose values are ``None``."""
    return {k: v for k, v in d.items() if v is not None}


def normalize_node(node: GraphNode) -> dict[str, Any]:
    """Explicitly construct a deterministic snapshot for a GraphNode."""
    out: dict[str, Any] = {
        "id": node.id,
        "kind": node.kind,
        "state": node.state,
    }
    tokens = getattr(node, "tokens", None)
    if tokens is not None:
        out["tokens"] = tokens
    evidence = getattr(node, "evidence", None)
    if evidence is not None:
        out["evidence"] = {
            "snippet": evidence.snippet,
            **({"file": evidence.file} if evidence.file is not None else {}),
            **({"startLine": evidence.startLine} if evidence.startLine is not None else {}),
        }
    return out


def normalize_edge(edge: GraphEdge) -> dict[str, Any]:
    """Explicitly construct a deterministic snapshot for a GraphEdge."""
    out: dict[str, Any] = {
        "source": edge.source,
        "target": edge.target,
        "kind": edge.kind,
    }
    if hasattr(edge, "primary") and edge.primary is not None:
        out["primary"] = edge.primary
    return out


def normalize_graph(graph: Any) -> dict[str, list[dict[str, Any]]]:
    """Normalize a NetworkX DiGraph into a deterministic structure."""
    nodes: list[dict[str, Any]] = []
    for _, data in graph.nodes(data=True):
        if isinstance(data, dict):
            # If the node stores a typed GraphNode under the "data" key, use it.
            if "data" in data:
                node_obj = cast(GraphNode, data["data"])
                normalized = normalize_node(node_obj)  # type: ignore[arg-type]
            else:
                # Ensure required fields exist for raw dict nodes.
                for k in ("id", "kind", "state"):
                    if k not in data:
                        raise ValueError(f"Node missing required field '{k}'")
                tmp = SimpleNamespace(**data)
                normalized = normalize_node(tmp)  # type: ignore[arg-type]
        else:
            normalized = normalize_node(data)  # type: ignore[arg-type]
        nodes.append(normalized)

    edges: list[dict[str, Any]] = []
    for source, target, data in graph.edges(data=True):
        if isinstance(data, dict):
            # If a typed GraphEdge is stored under the "data" key, use it directly.
            if "data" in data:
                edge_obj = cast(GraphEdge, data["data"])
                edge_dict = normalize_edge(edge_obj)  # type: ignore[arg-type]
            else:
                # Merge source and target into the temporary namespace for raw dicts.
                tmp = SimpleNamespace(source=source, target=target, **data)
                edge_dict = normalize_edge(tmp)  # type: ignore[arg-type]
        else:
            edge_dict = normalize_edge(data)
        edges.append(_remove_none(edge_dict))

    nodes.sort(key=lambda n: n["id"])
    edges.sort(key=lambda e: (e["source"], e["target"], e["kind"]))
    return {"nodes": nodes, "edges": edges}


def render_steps(steps: list[dict[str, Any]]) -> str:
    """Serialize a list of steps (event + graph) for snapshot testing."""
    rendered_steps = json.dumps(
        [
            {
                "event": normalize_event(step["event"]) if step["event"] else None,
                "graph": normalize_graph(step["graph"]),  # type: ignore[arg-type]
            }
            for step in steps
        ],
        sort_keys=True,
        indent=2,
    )
    return rendered_steps


def normalize_event(event: GraphEvent) -> dict[str, Any]:
    """Normalize a ``GraphEvent`` into a concise, deterministic dict."""
    if isinstance(event, AddNodesEvent):
        return {
            "type": "addNodes",
            "nodes": sorted([node.id for node in event.nodes]),
        }

    if isinstance(event, AddEdgesEvent):
        edges = [{"source": e.source, "target": e.target, "kind": e.kind} for e in event.edges]
        edges.sort(key=lambda d: (d["source"], d["target"], d["kind"]))
        return {"type": "addEdges", "edges": edges}

    if isinstance(event, UpdateNodeEvent):
        patch = _remove_none(event.patch)
        return {"type": "updateNode", "id": event.id, "patch": patch}

    return {"type": "iteration", "step": event.step}


def to_event_dict(event: GraphEvent) -> dict[str, Any]:
    """Return a JSON-serializable representation of a GraphEvent."""
    return normalize_event(event)


class GraphSnapshot:
    """Wrapper around the ``snapshot`` fixture providing domain‑aware assertions."""

    def __init__(self, snapshot: Snapshot) -> None:  # type: ignore[assignment]
        self._snapshot = snapshot
        self._counter = 0

    def _next_name(self, prefix: str, identifier: str | None = None) -> str:
        """Generate a deterministic snapshot name."""
        if prefix in ("node", "edge", "events"):
            self._counter += 1
            return f"{prefix}_{self._counter}"
        if prefix == "graph":
            return "graph"
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def _assert(self, data: Any, name: str) -> None:
        """Serialize *data* deterministically and forward to the underlying fixture."""
        rendered = json.dumps(data, sort_keys=True, indent=2)
        self._snapshot.assert_match(rendered, name)

    def assert_graph(self, graph: Graph) -> None:  # type: ignore[arg-type]
        self._assert(normalize_graph(graph), self._next_name("graph"))

    def assert_events(self, events: list[GraphEvent]) -> None:  # type: ignore[arg-type]
        normalized = [to_event_dict(ev) for ev in events]
        self._assert(normalized, self._next_name("events"))

    def assert_node(self, node: GraphNode) -> None:  # type: ignore[arg-type]
        self._assert(normalize_node(node), self._next_name("node", node.id))

    def assert_edge(self, edge: GraphEdge) -> None:  # type: ignore[arg-type]
        self._assert(normalize_edge(edge), self._next_name("edge", f"{edge.source}_{edge.target}"))


__all__ = [
    "normalize_node",
    "normalize_edge",
    "normalize_graph",
    "render_steps",
    "normalize_event",
    "to_event_dict",
    "GraphSnapshot",
]
