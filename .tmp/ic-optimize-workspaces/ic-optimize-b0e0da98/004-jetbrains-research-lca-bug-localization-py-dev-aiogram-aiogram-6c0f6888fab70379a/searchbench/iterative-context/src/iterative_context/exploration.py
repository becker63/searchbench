# pyright: reportUnusedFunction=false

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Literal, TypedDict, cast

from iterative_context.graph_models import Graph, GraphEdge, GraphNode, PendingNode
from iterative_context.injest.llm_tldr_adapter import ingest_repo
from iterative_context.normalize import raw_tree_to_graph
from iterative_context.scoring import default_score_fn
from iterative_context.serialization import serialize_graph
from iterative_context.store import GraphStore
from iterative_context.traversal import DefaultExpansionPolicy, run_traversal
from iterative_context.types import SelectionCallable


class GraphSnapshotDict(TypedDict):
    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]
    metadata: dict[str, object]

_active_graph: Graph | None = None
_active_store: GraphStore | None = None
_graph_signature: str | None = None
_active_repo_root: Path | None = None


def _set_active_graph(  # noqa: PLW0603
    graph: Graph, repo_root: Path | None = None, signature: str | None = None
) -> None:
    """
    Replace the active graph and its store.

    Optional repo metadata keeps the graph binding explicit when available.
    """
    global _active_graph, _active_store, _graph_signature, _active_repo_root  # noqa: PLW0603
    _active_graph = graph
    _active_store = GraphStore(graph)
    _active_repo_root = repo_root
    _graph_signature = signature


def _clear_active_graph() -> None:  # noqa: PLW0603
    """Reset the active graph and associated metadata."""
    global _active_graph, _active_store, _graph_signature, _active_repo_root  # noqa: PLW0603
    _active_graph = None
    _active_store = None
    _graph_signature = None
    _active_repo_root = None


def get_active_graph() -> Graph | None:
    """Return the currently loaded graph, if any."""
    return _active_graph


def _repo_signature(root: Path) -> str:
    """Compute a deterministic signature of Python sources under root."""
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(str(stat.st_mtime_ns).encode())
        hasher.update(str(stat.st_size).encode())
    return hasher.hexdigest()


def _normalize_kind(raw: str | None) -> Literal["symbol", "function", "file", "type"]:
    if raw == "function":
        return "function"
    if raw == "file":
        return "file"
    if raw == "type":
        return "type"
    return "symbol"


def _normalize_edge_kind(raw: str | None) -> Literal["calls", "imports", "references"]:
    if raw == "imports":
        return "imports"
    if raw == "calls":
        return "calls"
    return "references"


def _build_graph_from_repo(root: Path) -> Graph:
    raw_tree = ingest_repo(root)
    base_graph = raw_tree_to_graph(raw_tree)

    graph = Graph()
    for node_id, raw_data in base_graph.nodes(data=True):
        symbol_value: str | None = None
        file_value: str | None = None
        kind_value: str | None = None
        if isinstance(raw_data, dict):
            typed_data = cast(dict[str, object], raw_data)
            symbol_raw = typed_data.get("symbol")
            if isinstance(symbol_raw, str):
                symbol_value = symbol_raw
            file_raw = typed_data.get("file")
            if isinstance(file_raw, str):
                file_value = file_raw
            kind_raw = typed_data.get("type")
            kind_value = kind_raw if isinstance(kind_raw, str) else None

        node = PendingNode(id=str(node_id), kind=_normalize_kind(kind_value))
        attrs: dict[str, object] = {"data": node}
        if isinstance(symbol_value, str):
            attrs["symbol"] = symbol_value
        if isinstance(file_value, str):
            attrs["file"] = file_value
        graph.add_node(node.id, **attrs)

    for src, dst, raw_data in base_graph.edges(data=True):
        typed_edge = cast(dict[str, object], raw_data)
        embedded = typed_edge.get("data")
        if isinstance(embedded, GraphEdge):
            edge_kind = embedded.kind
        else:
            raw_kind = typed_edge.get("kind")
            edge_kind = raw_kind if isinstance(raw_kind, str) else None
        kind = _normalize_edge_kind(edge_kind)
        edge = GraphEdge(source=str(src), target=str(dst), kind=kind)
        graph.add_edge(edge.source, edge.target, data=edge)

    return graph


def _normalize_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is None:
        return Path.cwd().resolve()
    return Path(repo_root).resolve()


def ensure_graph_loaded(repo_root: str | Path | None = None) -> None:
    """
    Idempotently load the active graph from a repository.

    If repo_root is omitted, falls back to the current working directory.
    """
    root = _normalize_repo_root(repo_root)
    signature = _repo_signature(root)

    if (
        _active_graph is not None
        and _active_repo_root is not None
        and _active_repo_root == root
        and signature == _graph_signature
    ):
        return

    graph = _build_graph_from_repo(root)
    graph.graph["repo_root"] = str(root)
    graph.graph["repo_signature"] = signature
    _set_active_graph(graph, repo_root=root, signature=signature)


def get_active_repo_root() -> Path | None:
    return _active_repo_root


def get_active_repo_metadata() -> dict[str, object]:
    if _active_repo_root is None or _graph_signature is None:
        return {}
    return {"repo_root": str(_active_repo_root), "repo_signature": _graph_signature}


def resolve(symbol: str) -> GraphNode | None:
    """Public resolve: pure lookup via the active store."""
    if _active_store is None:
        return None
    return _active_store.resolve(symbol)


def expand(
    node_id: str,
    depth: int = 1,
    score_fn: SelectionCallable | None = None,
) -> GraphSnapshotDict:
    """
    Deterministic bounded expansion using internal traversal.

    Returns a normalized GraphSnapshot (dict) of the explored subgraph.
    """
    if _active_graph is None:
        return cast(
            GraphSnapshotDict,
            {"nodes": [], "edges": [], "metadata": {"depth": depth, "expanded_from": node_id}},
        )

    # Limit traversal to the connected neighborhood to avoid unrelated drift.
    base_reachable: set[str] = set()
    queue: list[tuple[str, int]] = [(node_id, 0)]
    while queue:
        current, dist = queue.pop(0)
        if current in base_reachable:
            continue
        base_reachable.add(current)
        if dist >= depth:
            continue
        for succ in _active_graph.successors(current):
            queue.append((cast(str, succ), dist + 1))
        for pred in _active_graph.predecessors(current):
            queue.append((cast(str, pred), dist + 1))

    working = cast(Graph, deepcopy(_active_graph.subgraph(base_reachable).copy()))
    steps = max(depth, 0)

    run_traversal(
        working,
        steps=steps,
        expansion_policy=DefaultExpansionPolicy(),
        score_fn=score_fn or default_score_fn,
    )

    visited: set[str] = set()
    frontier: list[tuple[str, int]] = [(node_id, 0)]
    while frontier:
        current, dist = frontier.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if dist >= steps:
            continue
        for succ in working.successors(current):
            frontier.append((cast(str, succ), dist + 1))
        for pred in working.predecessors(current):
            frontier.append((cast(str, pred), dist + 1))

    subgraph = cast(Graph, working.subgraph(visited).copy())
    return cast(
        GraphSnapshotDict,
        serialize_graph(subgraph, metadata={"depth": depth, "expanded_from": node_id}),
    )


def expand_with_policy(node_id: str, depth: int, score_fn: SelectionCallable) -> GraphSnapshotDict:
    return expand(node_id=node_id, depth=depth, score_fn=score_fn)


def resolve_and_expand(
    symbol: str,
    depth: int = 1,
    score_fn: SelectionCallable | None = None,
) -> GraphSnapshotDict:
    """Composition only: resolve first, then expand deterministically."""
    node = resolve(symbol)
    if node is None:
        return cast(GraphSnapshotDict, {"nodes": [], "edges": [], "metadata": {}})
    return expand(node.id, depth=depth, score_fn=score_fn)
