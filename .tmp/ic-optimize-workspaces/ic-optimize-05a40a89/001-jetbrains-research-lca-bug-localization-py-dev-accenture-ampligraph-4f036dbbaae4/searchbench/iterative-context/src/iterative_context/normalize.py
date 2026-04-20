from collections.abc import Iterable

from iterative_context.graph_models import Graph
from iterative_context.raw_tree import RawTree


def raw_tree_to_graph(tree: RawTree) -> Graph:
    """Create a minimal graph from a RawTree."""
    graph = Graph()

    def ensure_node(node_id: str, *, node_type: str, symbol: str, file: str | None) -> None:
        if node_id not in graph.nodes:
            graph.add_node(node_id, type=node_type, symbol=symbol, file=file)
        else:
            # Preserve existing attributes but backfill symbol/type/file if missing.
            data = graph.nodes[node_id]
            data.setdefault("type", node_type)
            data.setdefault("symbol", symbol)
            if file is not None:
                data.setdefault("file", file)

    for raw_file in sorted(tree.files, key=lambda f: f.path):
        for fn in raw_file.functions:
            ensure_node(fn.id, node_type="function", symbol=fn.name, file=fn.file)

    def iter_edges() -> Iterable[tuple[str, str, str, str]]:
        for raw_file in sorted(tree.files, key=lambda f: f.path):
            for edge in sorted(raw_file.edges, key=lambda e: (e.kind, e.source, e.target)):
                yield edge.kind, edge.source, edge.target, raw_file.path

    for kind, source, target, file_path in iter_edges():
        if kind == "import":
            ensure_node(source, node_type="unknown", symbol=source, file=file_path)
            ensure_node(target, node_type="module", symbol=target, file=None)
            graph.add_edge(source, target, kind="imports")
        elif kind == "call":
            ensure_node(source, node_type="unknown", symbol=source, file=file_path)
            ensure_node(target, node_type="unknown", symbol=target, file=None)
            graph.add_edge(source, target, kind="calls")

    return graph
