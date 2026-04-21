# pyright: reportMissingTypeStubs=false
from __future__ import annotations

from typing import cast

from hypothesis import given
from hypothesis import strategies as st

from iterative_context.expansion import expand_node  # type: ignore
from iterative_context.test_helpers import (
    EventSpec,
    GraphSpec,
    GraphType,
    apply_events,
    build_graph,
    normalize_graph,
)


def assert_graph_valid(graph: GraphType) -> None:
    allowed_states = {"pending", "resolved", "anchor", "pruned"}
    normalized = normalize_graph(graph)

    node_ids: list[str] = []
    for node in normalized["nodes"]:
        node_id = node["id"]
        assert node_id not in node_ids
        node_ids.append(node_id)
        assert node["state"] in allowed_states

    nodes_seen = set(node_ids)

    edges_seen: set[tuple[str, str, str]] = set()
    for edge in normalized["edges"]:
        key = (edge["source"], edge["target"], edge["kind"])
        assert key not in edges_seen
        edges_seen.add(key)

        assert edge["source"] in nodes_seen
        assert edge["target"] in nodes_seen


node_ids: st.SearchStrategy[str] = st.text(min_size=1, max_size=3)


def make_edge_spec(s: str, t: str, k: str) -> dict[str, str]:
    return {"source": s, "target": t, "kind": k}


def make_add_nodes(ns: list[str]) -> EventSpec:
    return cast(EventSpec, {"type": "addNodes", "nodes": ns})


def make_add_edges(edges: list[dict[str, str]]) -> EventSpec:
    return cast(EventSpec, {"type": "addEdges", "edges": edges})


def make_update_node(nid: str) -> EventSpec:
    return cast(
        EventSpec,
        {"type": "updateNode", "id": nid, "patch": {"state": "resolved", "tokens": 1}},
    )


edge_spec_strategy: st.SearchStrategy[dict[str, str]] = st.builds(
    make_edge_spec,
    node_ids,
    node_ids,
    st.sampled_from(["calls", "imports", "references"]),
)
event_strategy: st.SearchStrategy[EventSpec] = st.one_of(
    st.builds(
        make_add_nodes,
        st.lists(node_ids, min_size=1, max_size=3, unique=True),
    ),
    st.builds(
        make_add_edges,
        st.lists(edge_spec_strategy, min_size=1, max_size=3),
    ),
    st.builds(
        make_update_node,
        node_ids,
    ),
    st.just(cast(EventSpec, {"type": "iteration", "step": 0})),
)
event_sequences: st.SearchStrategy[list[EventSpec]] = st.lists(event_strategy, max_size=6)


@given(event_sequences)
def test_graph_always_valid(events: list[EventSpec]) -> None:
    base_spec: GraphSpec = {"nodes": [], "edges": []}
    graph = build_graph(base_spec)
    for event in events:
        try:
            apply_events(graph, [event])
        except ValueError:
            break
    assert_graph_valid(graph)


def test_expand_node_deterministic() -> None:
    node = {"id": "A", "state": "pending"}
    graph = build_graph({"nodes": [{"id": "A", "kind": "symbol", "state": "pending"}], "edges": []})
    events1 = expand_node(node, graph)  # type: ignore[arg-type]
    events2 = expand_node(node, graph)  # type: ignore[arg-type]
    assert events1 == events2


def test_expand_node_minimal_when_child_exists() -> None:
    node = {"id": "A", "state": "pending"}
    graph = build_graph(
        {
            "nodes": [
                {"id": "A", "kind": "symbol", "state": "pending"},
                {"id": "A_child", "kind": "symbol", "state": "pending"},
            ]
        }
    )
    events = expand_node(node, graph)  # type: ignore[arg-type]
    assert len(events) == 1
    assert events[0].type == "updateNode"


def test_expand_node_idempotent_effect() -> None:
    graph_spec: GraphSpec = {
        "nodes": [{"id": "A", "kind": "symbol", "state": "pending"}],
        "edges": [],
    }
    graph = build_graph(graph_spec)

    events1 = expand_node({"id": "A", "state": "pending"}, graph)  # type: ignore[arg-type]
    apply_events(graph, events1)  # type: ignore[arg-type]

    events2 = expand_node({"id": "A", "state": "pending"}, graph)  # type: ignore[arg-type]
    apply_events(graph, events2)  # type: ignore[arg-type]

    normalized = normalize_graph(graph)
    node_ids_seen = [n["id"] for n in normalized["nodes"]]
    assert len(node_ids_seen) == len(set(node_ids_seen))

    edge_tuples = [(e["source"], e["target"], e["kind"]) for e in normalized["edges"]]
    assert len(edge_tuples) == len(set(edge_tuples))


def test_expand_node_event_order() -> None:
    graph = build_graph({"nodes": [{"id": "A", "kind": "symbol", "state": "pending"}], "edges": []})
    events = expand_node({"id": "A", "state": "pending"}, graph)  # type: ignore[arg-type]
    assert [e.type for e in events] == ["addNodes", "addEdges", "updateNode"]


def test_roundtrip_stability() -> None:
    spec: GraphSpec = {
        "nodes": [
            {"id": "A", "kind": "symbol", "state": "pending"},
            {"id": "B", "kind": "symbol", "state": "pending"},
        ],
        "edges": [{"source": "A", "target": "B", "kind": "calls"}],
    }
    g1 = build_graph(spec)

    events: list[EventSpec] = [
        {"type": "addNodes", "nodes": ["A", "B"]},
        {"type": "addEdges", "edges": [{"source": "A", "target": "B", "kind": "calls"}]},
    ]

    g2 = build_graph(cast(GraphSpec, {"nodes": [], "edges": []}))
    apply_events(g2, events)

    assert normalize_graph(g1) == normalize_graph(g2)


@given(event_sequences)
def test_deterministic_replay(events: list[EventSpec]) -> None:
    empty_spec: GraphSpec = {"nodes": [], "edges": []}
    g1 = build_graph(empty_spec)
    g2 = build_graph(empty_spec)

    for event in events:
        try:
            apply_events(g1, [event])
            apply_events(g2, [event])
        except ValueError:
            break

    assert normalize_graph(g1) == normalize_graph(g2)
