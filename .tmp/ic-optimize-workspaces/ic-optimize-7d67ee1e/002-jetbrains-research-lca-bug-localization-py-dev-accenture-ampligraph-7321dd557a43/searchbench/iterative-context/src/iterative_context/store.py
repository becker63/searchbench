from collections import deque
from typing import Any, cast

from pydantic import BaseModel

from iterative_context.graph_models import Graph, GraphEdge, GraphNode


class GraphStore:
    """
    READ-ONLY PROJECTION LAYER.

    This class provides indexed, queryable access to the graph.

    It MUST NOT:
    - mutate the graph
    - emit events
    - perform expansion

    It only answers questions about the current graph state.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

        self.nodes_by_id: dict[str, dict[str, Any]] = {}
        self.nodes_by_symbol: dict[str, list[str]] = {}
        self.nodes_by_file: dict[str, list[str]] = {}
        self.index_by_symbol: dict[str, list[str]] = {}
        self.index_by_file: dict[str, list[str]] = {}

        self.out_edges: dict[str, list[GraphEdge]] = {}
        self.in_edges: dict[str, list[GraphEdge]] = {}

        self._build_indexes()

    def _build_indexes(self) -> None:
        for node_id, data in sorted(self.graph.nodes(data=True), key=lambda item: item[0]):
            node_data = dict(data)
            self.nodes_by_id[node_id] = node_data

            symbol = node_data.get("symbol")
            if isinstance(symbol, str):
                self.nodes_by_symbol.setdefault(symbol, []).append(node_id)
                self.index_by_symbol.setdefault(symbol, []).append(node_id)

            file_value = node_data.get("file")
            if isinstance(file_value, str):
                self.nodes_by_file.setdefault(file_value, []).append(node_id)
                self.index_by_file.setdefault(file_value, []).append(node_id)
        for mapping in (
            self.nodes_by_symbol,
            self.nodes_by_file,
            self.index_by_symbol,
            self.index_by_file,
        ):
            for key in mapping:
                mapping[key] = sorted(mapping[key])

        for src, dst, data in sorted(
            self.graph.edges(data=True), key=lambda item: (item[0], item[1])
        ):
            kind_value = data.get("kind")
            if kind_value not in {"calls", "imports", "references"}:
                data_obj = data.get("data")
                if isinstance(data_obj, GraphEdge):
                    kind_value = data_obj.kind
            kind = kind_value if isinstance(kind_value, str) else "references"

            edge = GraphEdge(
                source=src,
                target=dst,
                kind=kind,  # type: ignore[arg-type]
                id=data.get("id"),
                primary=data.get("primary"),
            )

            self.out_edges.setdefault(src, []).append(edge)
            self.in_edges.setdefault(dst, []).append(edge)

    def find_symbol(self, symbol: str) -> list[str]:
        return list(self.nodes_by_symbol.get(symbol, []))

    def _node_for_id(self, node_id: str) -> GraphNode | None:
        data = self.graph.nodes.get(node_id)
        if data is None:
            return None
        if "data" in data:
            inner = data.get("data")
            if isinstance(inner, BaseModel):
                return cast(GraphNode, inner)
        if isinstance(data, BaseModel):
            return cast(GraphNode, data)
        return None

    def resolve(self, query: str) -> GraphNode | None:
        """
        Deterministic lookup using symbol/file indexes only.

        No traversal or mutation.
        """
        exact = self.index_by_symbol.get(query, [])
        if exact:
            node = self._node_for_id(sorted(exact)[0])
            if node:
                return node

        lowered = query.lower()
        for sym, ids in sorted(self.index_by_symbol.items(), key=lambda item: item[0]):
            if lowered in sym.lower():
                node = self._node_for_id(ids[0])
                if node:
                    return node

        for path, ids in sorted(self.index_by_file.items(), key=lambda item: item[0]):
            if lowered in path.lower():
                node = self._node_for_id(ids[0])
                if node:
                    return node

        return None

    def get_neighbors(self, node_id: str, kinds: list[str] | None = None) -> list[str]:
        neighbors: set[str] = set()

        for edge in self.out_edges.get(node_id, []):
            if kinds is None or edge.kind in kinds:
                neighbors.add(edge.target)

        for edge in self.in_edges.get(node_id, []):
            if kinds is None or edge.kind in kinds:
                neighbors.add(edge.source)

        return sorted(neighbors)

    def get_neighborhood(self, node_id: str, radius: int) -> set[str]:
        if radius < 0:
            return set()

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        queue.append((node_id, 0))
        visited.add(node_id)

        while queue:
            current, dist = queue.popleft()
            if dist >= radius:
                continue

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return visited


def debug_summary(store: GraphStore) -> dict[str, int]:
    return {
        "nodes": len(store.nodes_by_id),
        "edges": sum(len(v) for v in store.out_edges.values()),
    }
