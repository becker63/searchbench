# pyright: reportPrivateUsage=false

import pytest

from iterative_context.exploration import _set_active_graph, expand
from iterative_context.graph_models import Graph
from iterative_context.store import GraphStore
from iterative_context.test_helpers.graph_dsl import build_graph


@pytest.fixture
def graph_fixture() -> Graph:
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
    _set_active_graph(graph)
    return graph


def test_neighborhood_does_not_mutate(graph_fixture: Graph) -> None:
    graph = graph_fixture
    store = GraphStore(graph)

    before_nodes = set(graph.nodes)
    before_edges = set(graph.edges)

    store.get_neighborhood(next(iter(graph.nodes)), radius=2)

    assert set(graph.nodes) == before_nodes
    assert set(graph.edges) == before_edges


def test_expand_grows_graph(graph_fixture: Graph) -> None:
    result = expand(next(iter(graph_fixture.nodes)), depth=2)

    assert result["nodes"]
