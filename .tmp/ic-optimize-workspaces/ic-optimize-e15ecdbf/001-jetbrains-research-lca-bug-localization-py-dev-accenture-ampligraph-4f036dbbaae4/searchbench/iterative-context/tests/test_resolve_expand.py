# pyright: reportPrivateUsage=false

import pytest

from iterative_context.exploration import (
    _set_active_graph,
    expand,
    expand_with_policy,
    resolve,
    resolve_and_expand,
)
from iterative_context.graph_models import Graph, GraphNode
from iterative_context.test_helpers.graph_dsl import build_graph


def make_graph_with_symbols() -> Graph:
    graph = build_graph(
        {
            "nodes": [
                {"id": "A", "kind": "symbol", "state": "pending"},
                {"id": "B", "kind": "symbol", "state": "pending"},
            ],
            "edges": [
                {"source": "A", "target": "B", "kind": "calls"},
            ],
        }
    )
    graph.nodes["A"]["symbol"] = "expand_node"
    graph.nodes["A"]["file"] = "src/iterative_context/expansion.py"
    graph.nodes["B"]["symbol"] = "other_symbol"
    graph.nodes["B"]["file"] = "src/iterative_context/other.py"
    return graph


@pytest.fixture
def activate_graph() -> Graph:
    graph = make_graph_with_symbols()
    _set_active_graph(graph)
    return graph


def test_resolve_symbol(activate_graph: Graph) -> None:
    node = resolve("expand_node")
    assert node is not None
    assert node.id == "A"


def test_resolve_deterministic(activate_graph: Graph) -> None:
    node1 = resolve("expand_node")
    node2 = resolve("expand_node")
    assert node1 is not None and node2 is not None
    assert node1.id == node2.id


def test_expand_returns_snapshot(activate_graph: Graph) -> None:
    snapshot = expand("A", depth=2)
    assert snapshot["nodes"]


def test_resolve_and_expand_integration(activate_graph: Graph) -> None:
    direct_node = resolve("expand_node")
    assert direct_node is not None
    manual = expand(direct_node.id, depth=2)

    combined = resolve_and_expand("expand_node", depth=2)

    assert manual == combined


def test_expand_with_callable_policy(activate_graph: Graph) -> None:
    def score(node: GraphNode, graph: Graph, step: int) -> float:
        return 1.0

    result = expand_with_policy("A", depth=1, score_fn=score)

    assert result is not None
