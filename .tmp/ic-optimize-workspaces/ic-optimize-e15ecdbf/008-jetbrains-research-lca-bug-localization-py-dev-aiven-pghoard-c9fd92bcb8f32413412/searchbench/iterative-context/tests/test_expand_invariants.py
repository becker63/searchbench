"""Invariant tests for expand() output."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
from typing import Any

import hypothesis.strategies as st
from hypothesis import given

from iterative_context.exploration import _set_active_graph, expand
from iterative_context.test_helpers.graph_dsl import build_graph
from iterative_context.test_helpers.snapshot_graph import normalize_graph


def _make_chain(length: int) -> Any:
    nodes: list[dict[str, str]] = [
        {"id": chr(ord("A") + i), "kind": "symbol", "state": "pending"} for i in range(length)
    ]
    edges: list[dict[str, str]] = []
    for i in range(length - 1):
        edges.append({"source": nodes[i]["id"], "target": nodes[i + 1]["id"], "kind": "calls"})
    return {"nodes": nodes, "edges": edges}


@given(st.sampled_from([_make_chain(1), _make_chain(2), _make_chain(3)]))
def test_expand_deterministic(spec: Any) -> None:
    graph = build_graph(spec)  # type: ignore[arg-type]
    _set_active_graph(graph)

    out1 = expand(spec["nodes"][0]["id"], depth=2)
    out2 = expand(spec["nodes"][0]["id"], depth=2)

    assert out1 == out2


def test_expand_idempotent() -> None:
    spec = _make_chain(2)
    graph = build_graph(spec)  # type: ignore[arg-type]
    _set_active_graph(graph)

    before = normalize_graph(graph)
    _ = expand("A", depth=2)
    after = normalize_graph(graph)

    assert before == after


def test_expand_locality() -> None:
    spec = _make_chain(2)
    base_graph = build_graph(spec)  # type: ignore[arg-type]
    _set_active_graph(base_graph)
    base_snapshot = expand("A", depth=2)

    expanded_graph = copy.deepcopy(base_graph)
    expanded_graph.add_node("Z", kind="symbol", state="pending")
    _set_active_graph(expanded_graph)
    noisy_snapshot = expand("A", depth=2)

    assert base_snapshot == noisy_snapshot


def test_expand_respects_depth() -> None:
    spec = _make_chain(3)
    graph = build_graph(spec)  # type: ignore[arg-type]
    _set_active_graph(graph)

    depth_one = expand("A", depth=1)
    depth_two = expand("A", depth=2)

    nodes_one = {n["id"] for n in depth_one["nodes"]}
    nodes_two = {n["id"] for n in depth_two["nodes"]}

    assert nodes_two.issuperset(nodes_one)
    assert "C" not in nodes_one
    assert "C" in nodes_two


def test_expand_no_dangling_edges() -> None:
    spec = _make_chain(3)
    graph = build_graph(spec)  # type: ignore[arg-type]
    _set_active_graph(graph)

    snapshot = expand("A", depth=2)
    node_ids = {n["id"] for n in snapshot["nodes"]}
    for edge in snapshot["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids


def test_expand_reports_metadata() -> None:
    graph = build_graph(_make_chain(2))  # type: ignore[arg-type]
    _set_active_graph(graph)

    snapshot = expand("A", depth=3)

    metadata = snapshot.get("metadata", {})
    assert metadata["depth"] == 3
    assert metadata["expanded_from"] == "A"
