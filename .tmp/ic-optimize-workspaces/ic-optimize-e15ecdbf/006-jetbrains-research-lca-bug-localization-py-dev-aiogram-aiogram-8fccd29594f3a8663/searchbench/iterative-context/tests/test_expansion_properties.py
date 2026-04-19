# pyright: reportMissingTypeStubs=false
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from iterative_context.expansion import expand_node  # type: ignore
from iterative_context.test_helpers import GraphType, build_graph

# 🔧 knobs
MAX_NODE_ID_SIZE = 5
MAX_NODES = 5


# strategies
node_ids: st.SearchStrategy[str] = st.text(
    min_size=1,
    max_size=MAX_NODE_ID_SIZE,
)

existing_nodes: st.SearchStrategy[list[str]] = st.lists(
    node_ids,
    unique=True,
    max_size=MAX_NODES,
)


def _make_node(nid: str) -> dict[str, str]:
    return {"id": nid, "state": "pending"}


def _graph_with_nodes(node_ids: list[str]) -> GraphType:
    return build_graph(
        {"nodes": [{"id": nid, "kind": "symbol", "state": "pending"} for nid in node_ids]}
    )  # type: ignore[arg-type]


node_strategy: st.SearchStrategy[dict[str, str]] = st.builds(
    _make_node,
    node_ids,
)


@given(node_strategy, existing_nodes)
def test_expand_deterministic(node: dict[str, str], existing: list[str]):
    graph = _graph_with_nodes(existing + [node["id"]])
    ev1 = expand_node(node, graph)
    ev2 = expand_node(node, graph)
    assert ev1 == ev2


@given(node_ids)
def test_expand_idempotent(node_id: str):
    node: dict[str, str] = {"id": node_id, "state": "pending"}
    existing: list[str] = [node_id, f"{node_id}_child"]
    graph = _graph_with_nodes(existing)

    events = expand_node(node, graph)

    assert len(events) == 1
    assert events[0].type == "updateNode"


@given(node_strategy, existing_nodes)
def test_no_duplicate_nodes(node: dict[str, str], existing: list[str]):
    graph = _graph_with_nodes(existing + [node["id"]])
    events: list[Any] = expand_node(node, graph)

    added: set[str] = set()
    for e in events:
        if e.type == "addNodes":
            for n in e.nodes:
                assert n.id not in added
                added.add(n.id)


@given(node_strategy, existing_nodes)
def test_edges_valid(node: dict[str, str], existing: list[str]):
    graph = _graph_with_nodes(existing + [node["id"]])
    events: list[Any] = expand_node(node, graph)

    known: set[str] = {node["id"], *existing}

    for e in events:
        if e.type == "addNodes":
            for n in e.nodes:
                known.add(n.id)

        if e.type == "addEdges":
            for edge in e.edges:
                assert edge.source in known
                assert edge.target in known


@given(node_strategy, existing_nodes)
def test_event_ordering(node: dict[str, str], existing: list[str]):
    graph = _graph_with_nodes(existing + [node["id"]])
    events: list[Any] = expand_node(node, graph)

    seen_nodes: set[str] = {node["id"], *existing}

    for e in events:
        if e.type == "addNodes":
            for n in e.nodes:
                seen_nodes.add(n.id)

        if e.type == "addEdges":
            for edge in e.edges:
                assert edge.source in seen_nodes
                assert edge.target in seen_nodes


@given(node_strategy, existing_nodes)
def test_no_redundant_addnodes(node: dict[str, str], existing: list[str]):
    graph = _graph_with_nodes(existing + [node["id"]])
    events: list[Any] = expand_node(node, graph)

    added: set[str] = set()

    for e in events:
        if e.type == "addNodes":
            for n in e.nodes:
                assert n.id not in added
                added.add(n.id)


@given(node_ids)
def test_single_child(node_id: str):
    node: dict[str, str] = {"id": node_id, "state": "pending"}

    graph = _graph_with_nodes([node_id])

    events: list[Any] = expand_node(node, graph)

    created: list[str] = [n.id for e in events if e.type == "addNodes" for n in e.nodes]

    assert len(created) <= 1


@given(node_strategy, existing_nodes)
def test_update_node_always_present(node: dict[str, str], existing: list[str]):
    graph = _graph_with_nodes(existing + [node["id"]])
    events: list[Any] = expand_node(node, graph)

    assert any(e.type == "updateNode" for e in events)
