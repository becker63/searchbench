from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import networkx as nx

from harness.localization.models import canonicalize_path

from .raw_tree import RawEdge, RawTree


def _ensure_node(
    graph: nx.DiGraph[Any], node_id: str, *, node_type: str, symbol: str, file: str | None
) -> None:
    if node_id not in graph.nodes:
        graph.add_node(node_id, type=node_type, symbol=symbol, file=file)
    else:
        data = graph.nodes[node_id]
        data.setdefault("type", node_type)
        data.setdefault("symbol", symbol)
        if file is not None:
            data.setdefault("file", file)


def _iter_edges(tree: RawTree) -> Iterable[tuple[str, RawEdge]]:
    for raw_file in sorted(tree.files, key=lambda f: f.path):
        for edge in sorted(raw_file.edges, key=lambda e: (e.kind, e.source, e.target)):
            yield canonicalize_path(raw_file.path), edge


def raw_tree_to_graph(tree: RawTree) -> nx.DiGraph[Any]:
    """Create a minimal directed graph from a RawTree."""
    graph: nx.DiGraph[Any] = nx.DiGraph()

    for raw_file in sorted(tree.files, key=lambda f: f.path):
        file_id = canonicalize_path(raw_file.path)
        _ensure_node(graph, file_id, node_type="file", symbol=file_id, file=file_id)
        for fn in raw_file.functions:
            fn_id = canonicalize_path(fn.id)
            _ensure_node(
                graph,
                fn_id,
                node_type="function",
                symbol=fn.name,
                file=canonicalize_path(fn.file),
            )
            # Connect function to file (declared-in/contains) if not already present.
            graph.add_edge(file_id, fn_id, kind="contains")

    for file_path, edge in _iter_edges(tree):
        source = canonicalize_path(edge.source)
        target = canonicalize_path(edge.target)
        kind = edge.kind
        if kind == "import":
            _ensure_node(graph, source, node_type="file", symbol=source, file=file_path)
            _ensure_node(graph, target, node_type="module", symbol=target, file=None)
            graph.add_edge(source, target, kind="imports")
        elif kind == "call":
            _ensure_node(graph, source, node_type="unknown", symbol=source, file=file_path)
            _ensure_node(graph, target, node_type="unknown", symbol=target, file=None)
            graph.add_edge(source, target, kind="calls")
        elif kind in {"contains", "declares"}:
            _ensure_node(graph, source, node_type="unknown", symbol=source, file=file_path)
            _ensure_node(graph, target, node_type="unknown", symbol=target, file=file_path)
            graph.add_edge(source, target, kind="contains")
        else:
            _ensure_node(graph, source, node_type="unknown", symbol=source, file=file_path)
            _ensure_node(graph, target, node_type="unknown", symbol=target, file=None)
            graph.add_edge(source, target, kind="references")

    return graph
