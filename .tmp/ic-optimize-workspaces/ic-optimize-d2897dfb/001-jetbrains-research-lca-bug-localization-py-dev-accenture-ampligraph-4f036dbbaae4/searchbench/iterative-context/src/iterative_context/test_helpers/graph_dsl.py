# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Deterministic, declarative helpers for building and mutating graphs in tests."""

# Hypothesis Guidelines:
#
# - Keep expand_node deterministic
# - Avoid randomness in selection functions
# - Prefer simple integer strategies (e.g., step counts)
# - Use normalize_graph for assertions
# - Let Hypothesis shrink failures naturally
#
# DO NOT:
# - introduce non-determinism
# - mutate global state
# - depend on ordering of dicts or sets

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NotRequired, TypedDict

import networkx as nx

from iterative_context.expansion import expand_node
from iterative_context.graph_models import (
    AddEdgesEvent,
    AddNodesEvent,
    AnchorNode,
    Evidence,
    Graph,
    GraphEdge,
    GraphEvent,
    GraphNode,
    IterationEvent,
    PendingNode,
    PrunedNode,
    ResolvedNode,
    UpdateNodeEvent,
)
from iterative_context.test_helpers.snapshot_graph import normalize_graph

type GraphType = Graph


class EvidenceSpec(TypedDict):
    snippet: str
    file: NotRequired[str | None]
    startLine: NotRequired[int | None]


class NodeSpec(TypedDict):
    id: str
    kind: Literal["symbol", "function", "file", "type"]
    state: Literal["pending", "pruned", "anchor", "resolved"]
    label: NotRequired[str | None]
    tokens: NotRequired[int | None]
    evidence: NotRequired[EvidenceSpec]


class EdgeSpec(TypedDict):
    source: str
    target: str
    kind: Literal["calls", "imports", "references"]
    id: NotRequired[str | None]
    primary: NotRequired[bool | None]


# ----------------------------------------------------------------------
# DSL Event specifications (TypedDicts)
# ----------------------------------------------------------------------


class AddNodesEventSpec(TypedDict):
    type: Literal["addNodes"]
    nodes: list[str]


class AddEdgesEventSpec(TypedDict):
    type: Literal["addEdges"]
    edges: list[EdgeSpec]


class UpdateNodeEventSpec(TypedDict):
    type: Literal["updateNode"]
    id: str
    patch: dict[str, Any]


class IterationEventSpec(TypedDict):
    type: Literal["iteration"]
    step: int


EventSpec = AddNodesEventSpec | AddEdgesEventSpec | UpdateNodeEventSpec | IterationEventSpec


class GraphSpec(TypedDict, total=False):
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]


# ----------------------------------------------------------------------
# Conversion from DSL EventSpec to runtime GraphEvent
# ----------------------------------------------------------------------


def to_graph_event(event: EventSpec) -> GraphEvent:
    """Convert a DSL EventSpec (TypedDict) into a runtime GraphEvent."""
    if event["type"] == "addNodes":
        nodes = [PendingNode(id=n, kind="symbol") for n in event["nodes"]]
        return AddNodesEvent(nodes=nodes)

    if event["type"] == "addEdges":
        edges = [
            GraphEdge(
                source=e["source"],
                target=e["target"],
                kind=e["kind"],
            )
            for e in event["edges"]
        ]
        return AddEdgesEvent(edges=edges)

    if event["type"] == "updateNode":
        return UpdateNodeEvent(id=event["id"], patch=event["patch"])

    if event["type"] == "iteration":
        return IterationEvent(step=event["step"])

    raise ValueError(f"Unknown event type: {event['type']}")


def _require_keys(label: str, obj: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ValueError(f"{label} missing required keys: {', '.join(missing)}")


def build_graph(dsl: GraphSpec) -> GraphType:
    """Construct a NetworkX DiGraph from a declarative DSL spec."""
    graph: GraphType = nx.DiGraph()

    for node in dsl.get("nodes", []):
        _require_keys("Node spec", node, ("id", "kind", "state"))
        # Determine which concrete GraphNode subclass to instantiate based on the ``state`` field.
        state = node["state"]
        if state == "pending":
            typed_node: GraphNode = PendingNode(id=node["id"], kind=node["kind"])
        elif state == "pruned":
            typed_node = PrunedNode(id=node["id"], kind=node["kind"])
        elif state == "anchor":
            evidence_spec = node.get("evidence")
            evidence = Evidence(**evidence_spec) if evidence_spec else None
            typed_node = AnchorNode(
                id=node["id"],
                kind=node["kind"],
                tokens=node.get("tokens"),
                evidence=evidence,
                label=node.get("label"),
            )
        elif state == "resolved":
            evidence_spec = node.get("evidence")
            if evidence_spec is None:
                raise ValueError("Resolved nodes must have evidence")
            evidence = Evidence(**evidence_spec)
            typed_node = ResolvedNode(
                id=node["id"],
                kind=node["kind"],
                tokens=node["tokens"],  # type: ignore[index]
                evidence=evidence,
                label=node.get("label"),
            )
        else:
            raise ValueError(f"Unsupported node state: {state}")
        graph.add_node(typed_node.id, data=typed_node)

    for edge in dsl.get("edges", []):
        _require_keys("Edge spec", edge, ("source", "target", "kind"))
        if edge["source"] not in graph.nodes or edge["target"] not in graph.nodes:
            raise ValueError(f"Edge references unknown nodes: {edge['source']} -> {edge['target']}")

        # Construct a typed GraphEdge object.
        edge_obj = GraphEdge(
            source=edge["source"],
            target=edge["target"],
            kind=edge["kind"],
            id=edge.get("id"),
            primary=edge.get("primary"),
        )
        graph.add_edge(edge["source"], edge["target"], data=edge_obj)

    return graph


def apply_events(graph: GraphType, events: Sequence[EventSpec | GraphEvent]) -> GraphType:
    """Apply a sequence of events (DSL EventSpec or runtime GraphEvent) to mutate the graph."""
    for raw_event in events:
        event = raw_event
        # If the event is a DSL spec (TypedDict), convert it to a GraphEvent first.
        if not isinstance(event, (AddNodesEvent, AddEdgesEvent, UpdateNodeEvent, IterationEvent)):
            event = to_graph_event(event)  # type: ignore[assignment]

        if isinstance(event, AddNodesEvent):
            for node in event.nodes:
                if node.id in graph.nodes:
                    continue
                graph.add_node(node.id, data=node)
        elif isinstance(event, AddEdgesEvent):
            for edge in event.edges:
                if edge.source not in graph.nodes or edge.target not in graph.nodes:
                    raise ValueError(
                        f"Edge references unknown nodes: {edge.source} -> {edge.target}"
                    )
            for edge in event.edges:
                graph.add_edge(edge.source, edge.target, data=edge)
        elif isinstance(event, UpdateNodeEvent):
            node_id = event.id
            if node_id not in graph.nodes:
                raise ValueError(f"UpdateNode references unknown node: {node_id}")
            existing_node = graph.nodes[node_id]["data"]
            updated_node = existing_node.model_copy(update=event.patch)
            graph.nodes[node_id]["data"] = updated_node
        else:
            # IterationEvent: observational; no mutation.
            continue
    return graph


def replay_with_snapshots(
    graph: GraphType, events: Sequence[GraphEvent | EventSpec]
) -> list[GraphType]:
    """Apply events stepwise, returning graph copies after each step (including initial)."""
    snapshots: list[GraphType] = [copy.deepcopy(graph)]
    for event in events:
        apply_events(graph, [event])
        snapshots.append(copy.deepcopy(graph))
    return snapshots


def replay_with_event_snapshots(
    graph: GraphType, events: Sequence[EventSpec | GraphEvent]
) -> list[dict[str, GraphType | GraphEvent | None]]:
    """Apply events stepwise, returning event-state pairs (initial state included)."""
    snapshots: list[dict[str, GraphType | GraphEvent | None]] = [
        {"event": None, "graph": copy.deepcopy(graph)}
    ]
    for event in events:
        if isinstance(event, (AddNodesEvent, AddEdgesEvent, UpdateNodeEvent, IterationEvent)):
            converted: GraphEvent = event
        else:
            converted = to_graph_event(event)

        apply_events(graph, [converted])
        snapshots.append({"event": converted, "graph": copy.deepcopy(graph)})
    return snapshots


def run_expansion_steps(
    initial_graph: GraphType,
    max_steps: int,
    select_node_fn: Callable[[GraphType], str],
) -> tuple[GraphType, list[GraphEvent]]:
    graph = initial_graph
    events: list[GraphEvent] = []

    max_nodes_threshold = 50
    for _ in range(max_steps):
        if len(graph.nodes) > max_nodes_threshold:
            break

        node_id = select_node_fn(graph)

        step_events = expand_node(
            {"id": node_id, "state": "pending"},
            graph,
        )

        events.extend(step_events)
        graph = apply_events(graph, step_events)

    return graph, events


def select_first_node(graph: GraphType) -> str:
    return sorted(graph.nodes)[0]


def graphs_equal(g1: GraphType, g2: GraphType) -> bool:
    return normalize_graph(g1) == normalize_graph(g2)


def debug_run(
    initial_graph: GraphType, steps: int, select_node_fn: Callable[[GraphType], str]
):
    graph, events = run_expansion_steps(initial_graph, steps, select_node_fn)
    return {
        "final_graph": normalize_graph(graph),
        "event_count": len(events),
    }


__all__ = [
    "build_graph",
    "apply_events",
    "replay_with_snapshots",
    "replay_with_event_snapshots",
    "GraphSpec",
    "GraphEvent",
    "EventSpec",
    "GraphType",
]
