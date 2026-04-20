import pytest

from iterative_context.graph_models import Graph, GraphEdge
from iterative_context.injest.llm_tldr_adapter import ingest_repo
from iterative_context.normalize import raw_tree_to_graph
from iterative_context.raw_tree import RawTree
from iterative_context.store import GraphStore
from iterative_context.test_helpers.repos import get_repo


@pytest.fixture(params=["small", "medium"])
def store_fixture(request: pytest.FixtureRequest) -> tuple[str, RawTree, Graph, GraphStore]:
    name = request.param
    path = get_repo(name)
    tree = ingest_repo(path)
    graph = raw_tree_to_graph(tree)
    return name, tree, graph, GraphStore(graph)


def test_store_indexes_all_nodes(store_fixture: tuple[str, RawTree, Graph, GraphStore]) -> None:
    _, _, graph, store = store_fixture

    assert set(store.nodes_by_id) == set(graph.nodes)


def test_find_symbol_returns_function_ids(
    store_fixture: tuple[str, RawTree, Graph, GraphStore],
) -> None:
    name, tree, _, store = store_fixture

    fn = next((f.functions[0] for f in tree.files if f.functions), None)
    assert fn is not None, f"{name} repo has no functions"

    matches = store.find_symbol(fn.name)

    assert fn.id in matches


def test_neighbors_match_graph(store_fixture: tuple[str, RawTree, Graph, GraphStore]) -> None:
    _, _, graph, store = store_fixture

    for node_obj in graph.nodes:
        node = str(node_obj)
        expected: set[str] = set()
        for src_obj, dst_obj in graph.edges:
            src = str(src_obj)
            dst = str(dst_obj)
            if src == node:
                expected.add(dst)
            if dst == node:
                expected.add(src)

        neighbors: set[str] = set(store.get_neighbors(node))
        assert neighbors == expected


def test_neighborhood_radius_one(store_fixture: tuple[str, RawTree, Graph, GraphStore]) -> None:
    name, tree, _, store = store_fixture

    fn = next((f.functions[0] for f in tree.files if f.functions), None)
    assert fn is not None, f"{name} repo has no functions"

    expected: set[str] = set(store.get_neighbors(fn.id))
    expected.add(fn.id)

    assert store.get_neighborhood(fn.id, radius=1) == expected


def test_store_is_deterministic(store_fixture: tuple[str, RawTree, Graph, GraphStore]) -> None:
    _, _, graph, _ = store_fixture

    store_one = GraphStore(graph)
    store_two = GraphStore(graph)

    assert store_one.nodes_by_id == store_two.nodes_by_id
    assert store_one.nodes_by_symbol == store_two.nodes_by_symbol
    assert store_one.nodes_by_file == store_two.nodes_by_file

    def edge_signature(
        mapping: dict[str, list[GraphEdge]],
    ) -> list[tuple[str, str, str, str | None]]:
        sig: list[tuple[str, str, str, str | None]] = []
        for key in sorted(mapping):
            sig.extend((key, e.source, e.target, e.kind) for e in mapping[key])
        return sig

    assert edge_signature(store_one.out_edges) == edge_signature(store_two.out_edges)
    assert edge_signature(store_one.in_edges) == edge_signature(store_two.in_edges)
