from __future__ import annotations

from collections import deque
from typing import Any, Callable

import networkx as nx
from .raw_tree import RawTree


class GraphStore:
    """
    Read-only projection over a normalized DiGraph.
    Provides deterministic symbol/file lookup, neighbors, and hop-distance queries.
    """

    def __init__(self, graph: nx.DiGraph[Any]):
        self.graph = graph

        self.nodes_by_id: dict[str, dict[str, Any]] = {}
        self.nodes_by_symbol: dict[str, list[str]] = {}
        self.nodes_by_file: dict[str, list[str]] = {}

        self.out_edges: dict[str, list[dict[str, Any]]] = {}
        self.in_edges: dict[str, list[dict[str, Any]]] = {}

        self._build_indexes()

    @classmethod
    def from_raw_tree(
        cls, tree: RawTree, *, normalizer: Callable[[RawTree], nx.DiGraph[Any]]
    ) -> "GraphStore":
        graph = normalizer(tree)
        return cls(graph)

    def _build_indexes(self) -> None:
        for node_id, data in sorted(self.graph.nodes(data=True), key=lambda item: item[0]):
            node_data = dict(data)
            self.nodes_by_id[node_id] = node_data
            symbol = node_data.get("symbol")
            file_value = node_data.get("file")
            if isinstance(symbol, str):
                self.nodes_by_symbol.setdefault(symbol, []).append(node_id)
            if isinstance(file_value, str):
                self.nodes_by_file.setdefault(file_value, []).append(node_id)

        for mapping in (self.nodes_by_symbol, self.nodes_by_file):
            for key in mapping:
                mapping[key] = sorted(mapping[key])

        for src, dst, data in sorted(
            self.graph.edges(data=True), key=lambda item: (item[0], item[1])
        ):
            edge_data = dict(data)
            self.out_edges.setdefault(src, []).append(edge_data | {"target": dst})
            self.in_edges.setdefault(dst, []).append(edge_data | {"source": src})

    def find_symbol(self, symbol: str) -> list[str]:
        return list(self.nodes_by_symbol.get(symbol, []))

    def find_file(self, path: str) -> list[str]:
        return list(self.nodes_by_file.get(path, []))

    def get_neighbors(self, node_id: str) -> list[str]:
        neighbors: set[str] = set()
        for edge in self.out_edges.get(node_id, []):
            neighbors.add(edge["target"])
        for edge in self.in_edges.get(node_id, []):
            neighbors.add(edge["source"])
        return sorted(neighbors)

    def get_neighborhood(self, node_id: str, radius: int) -> set[str]:
        if radius < 0:
            return set()
        visited: set[str] = set([node_id])
        queue: deque[tuple[str, int]] = deque()
        queue.append((node_id, 0))
        while queue:
            current, dist = queue.popleft()
            if dist >= radius:
                continue
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return visited

    def shortest_hop(self, sources: list[str], target: str) -> int | None:
        """
        Minimal unweighted hop distance between any source and target treating the graph as undirected.
        """
        if not sources:
            return None
        undirected = self.graph.to_undirected(as_view=True)
        best: int | None = None
        for source in sources:
            if source not in undirected or target not in undirected:
                continue
            try:
                dist = int(nx.shortest_path_length(undirected, source=source, target=target))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if best is None or dist < best:
                best = dist
        return best


def debug_summary(store: GraphStore) -> dict[str, int]:
    return {
        "nodes": len(store.nodes_by_id),
        "edges": sum(len(v) for v in store.out_edges.values()),
    }
