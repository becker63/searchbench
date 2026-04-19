# pyright: reportPrivateUsage=false
from typing import Any

from iterative_context.expansion import expand_node  # type: ignore
from iterative_context.exploration import _set_active_graph, expand
from iterative_context.test_helpers import (  # type: ignore
    GraphSnapshot,
    apply_events,
    build_graph,
    render_steps,
    replay_with_event_snapshots,
)


def test_linear_expansion(snapshot_graph: GraphSnapshot, snapshot: Any):
    # initial graph with a single node "A"
    initial = build_graph({"nodes": [{"id": "A", "kind": "symbol", "state": "pending"}]})  # type: ignore
    graph = initial  # type: ignore
    all_events = []

    current = "A"
    # perform three expansions along a linear chain
    for _ in range(3):
        events = expand_node({"id": current, "state": "pending"}, graph)
        all_events.extend(events)  # type: ignore[arg-type]
        graph = apply_events(graph, events)  # type: ignore
        current = f"{current}_child"

    # replay and snapshot the step trace
    steps = replay_with_event_snapshots(initial, all_events)  # type: ignore
    rendered = render_steps(steps)  # type: ignore
    snapshot.assert_match(rendered, "linear_expansion_steps")


def test_expand_idempotent(snapshot_graph: GraphSnapshot):
    # graph already contains parent and its child
    graph = build_graph(  # type: ignore
        {
            "nodes": [
                {"id": "A", "kind": "symbol", "state": "pending"},
                {"id": "A_child", "kind": "symbol", "state": "pending"},
            ]
        }
    )
    events = expand_node({"id": "A", "state": "pending"}, graph)  # type: ignore
    final_graph = apply_events(graph, events)  # type: ignore
    snapshot_graph.assert_graph(final_graph)  # type: ignore


def test_expand_deterministic():
    node = {"id": "A", "state": "pending"}
    graph = build_graph({"nodes": [{"id": "A", "kind": "symbol", "state": "pending"}]})  # type: ignore
    ev1 = expand_node(node, graph)
    ev2 = expand_node(node, graph)
    assert ev1 == ev2


def test_no_duplicate_child():
    node = {"id": "A", "state": "pending"}
    graph = build_graph(
        {
            "nodes": [
                {"id": "A", "kind": "symbol", "state": "pending"},
                {"id": "A_child", "kind": "symbol", "state": "pending"},
            ]
        }
    )  # type: ignore[arg-type]
    events = expand_node(node, graph)
    assert len(events) == 1
    assert events[0].type == "updateNode"


def test_no_branching():
    node = {"id": "A", "state": "pending"}
    graph = build_graph({"nodes": [{"id": "A", "kind": "symbol", "state": "pending"}]})  # type: ignore[arg-type]
    events = expand_node(node, graph)
    add_nodes_event = next(e for e in events if e.type == "addNodes")
    assert len(add_nodes_event.nodes) == 1
    total_created = sum(len(e.nodes) for e in events if e.type == "addNodes")
    assert total_created == 1


def test_valid_edge_structure():
    node = {"id": "A", "state": "pending"}
    graph = build_graph({"nodes": [{"id": "A", "kind": "symbol", "state": "pending"}]})  # type: ignore[arg-type]
    events = expand_node(node, graph)
    add_edges = next(e for e in events if e.type == "addEdges")
    edge = add_edges.edges[0]
    assert edge.source == "A"
    assert edge.target == "A_child"
    assert edge.kind == "calls"


def test_expand_respects_depth():
    spec = {
        "nodes": [
            {"id": "A", "kind": "symbol", "state": "pending"},
            {"id": "B", "kind": "symbol", "state": "pending"},
            {"id": "C", "kind": "symbol", "state": "pending"},
        ],
        "edges": [
            {"source": "A", "target": "B", "kind": "calls"},
            {"source": "B", "target": "C", "kind": "calls"},
        ],
    }
    graph = build_graph(spec)  # type: ignore[arg-type]
    _set_active_graph(graph)

    depth_one = expand("A", depth=1)
    depth_two = expand("A", depth=2)

    assert len(depth_two["nodes"]) >= len(depth_one["nodes"])
