# pyright: reportMissingTypeStubs=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Tests for the deterministic graph DSL and snapshot integration."""

from pytest_snapshot.plugin import Snapshot

from iterative_context.test_helpers import (
    EventSpec,
    GraphSnapshot,
    GraphType,
    build_graph,
    render_steps,
    replay_with_event_snapshots,
)


def test_node_variants(snapshot_graph: GraphSnapshot):
    g: GraphType = build_graph({"nodes": [{"id": "n1", "kind": "symbol", "state": "pending"}]})
    snapshot_graph.assert_graph(g)


def test_edge_and_events(snapshot_graph: GraphSnapshot, snapshot: Snapshot):
    events: list[EventSpec] = [
        {"type": "addNodes", "nodes": ["n5"]},
        {"type": "addEdges", "edges": [{"source": "n1", "target": "n2", "kind": "calls"}]},
        {"type": "updateNode", "id": "n5", "patch": {"state": "resolved", "tokens": 3}},
        {"type": "iteration", "step": 1},
    ]
    g: GraphType = build_graph(
        {
            "nodes": [
                {"id": "n5", "kind": "symbol", "state": "pending"},
                {"id": "n1", "kind": "symbol", "state": "pending"},
                {"id": "n2", "kind": "symbol", "state": "pending"},
            ],
            "edges": [],
        }
    )
    steps = replay_with_event_snapshots(g, events)
    final_graph = steps[-1]["graph"]
    snapshot_graph.assert_graph(final_graph)  # type: ignore[arg-type]
    rendered_steps = render_steps(steps)
    snapshot.assert_match(rendered_steps, "event_graph_steps")
