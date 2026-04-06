from __future__ import annotations

from iterative_context.graph_models import Graph, GraphNode  # type: ignore[import-not-found]


def score(node: GraphNode, graph: Graph, step: int) -> float:
    """Deterministic scoring function aligned to SelectionCallable."""
    total: float = 0.0

    if node.state == "pending":
        total += 5.0
    if node.state == "anchor":
        total += 10.0

    degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    total += degree * 0.5

    if node.state in ("anchor", "resolved"):
        tokens_val = getattr(node, "tokens", None)
        if isinstance(tokens_val, (int, float)):
            total += min(tokens_val, 50) * 0.1

    return float(total)
