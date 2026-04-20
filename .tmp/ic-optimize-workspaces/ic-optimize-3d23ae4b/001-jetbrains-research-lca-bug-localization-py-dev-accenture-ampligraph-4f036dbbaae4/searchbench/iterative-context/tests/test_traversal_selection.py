# pyright: reportPrivateUsage=false

from typing import cast

import pytest

from iterative_context.graph_models import Graph, GraphNode
from iterative_context.scoring import default_score_fn
from iterative_context.test_helpers import build_graph, normalize_graph
from iterative_context.traversal import run_traversal
from iterative_context.types import SelectionCallable


def test_traversal_changes_with_policy() -> None:
    base_spec = {
        "nodes": [
            {"id": "A", "kind": "symbol", "state": "pending"},
            {"id": "B", "kind": "symbol", "state": "pending"},
        ],
        "edges": [],
    }

    graph_a = build_graph(base_spec)  # type: ignore[arg-type]
    graph_b = build_graph(base_spec)  # type: ignore[arg-type]

    def prefer_alpha_start(node: GraphNode, graph: Graph, step: int) -> float:
        return -float(ord(node.id[0]))

    def prefer_alpha_end(node: GraphNode, graph: Graph, step: int) -> float:
        return float(ord(node.id[0]))

    run_traversal(graph_a, steps=1, score_fn=prefer_alpha_start)
    run_traversal(graph_b, steps=1, score_fn=prefer_alpha_end)

    assert normalize_graph(graph_a) != normalize_graph(graph_b)


def test_policy_must_return_numeric() -> None:
    graph = build_graph({"nodes": [{"id": "A", "kind": "symbol", "state": "pending"}]})  # type: ignore[arg-type]

    def bad_policy(node: GraphNode, graph: Graph, step: int) -> object:
        return "not-a-number"

    with pytest.raises(ValueError):
        run_traversal(graph, steps=1, score_fn=cast(SelectionCallable, bad_policy))


def test_external_score_fn_overrides_default() -> None:
    base_spec = {
        "nodes": [
            {"id": "A", "kind": "symbol", "state": "pending"},
            {"id": "B", "kind": "symbol", "state": "pending"},
        ],
        "edges": [],
    }
    graph = build_graph(base_spec)  # type: ignore[arg-type]

    def prefer_b(node: GraphNode, graph: Graph, step: int) -> float:
        return 10.0 if node.id == "B" else 0.0

    run_traversal(graph, steps=1, score_fn=prefer_b)

    assert "B_child" in graph.nodes


def test_default_score_fn_still_applies() -> None:
    base_spec = {
        "nodes": [
            {"id": "A", "kind": "symbol", "state": "pending"},
        ],
        "edges": [],
    }
    graph_one = build_graph(base_spec)  # type: ignore[arg-type]
    graph_two = build_graph(base_spec)  # type: ignore[arg-type]

    run_traversal(graph_one, steps=1, score_fn=None)
    run_traversal(graph_two, steps=1, score_fn=default_score_fn)

    assert normalize_graph(graph_one) == normalize_graph(graph_two)


def test_determinism_with_callable() -> None:
    base_spec = {
        "nodes": [
            {"id": "A", "kind": "symbol", "state": "pending"},
            {"id": "B", "kind": "symbol", "state": "pending"},
        ],
        "edges": [],
    }
    graph_one = build_graph(base_spec)  # type: ignore[arg-type]
    graph_two = build_graph(base_spec)  # type: ignore[arg-type]

    def prefer_a(node: GraphNode, graph: Graph, step: int) -> float:
        return 1.0 if node.id == "A" else 0.5

    run_traversal(graph_one, steps=2, score_fn=prefer_a)
    run_traversal(graph_two, steps=2, score_fn=prefer_a)

    assert normalize_graph(graph_one) == normalize_graph(graph_two)
