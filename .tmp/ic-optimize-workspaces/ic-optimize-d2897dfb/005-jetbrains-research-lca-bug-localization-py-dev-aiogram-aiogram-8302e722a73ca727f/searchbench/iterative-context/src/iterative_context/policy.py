"""Local scoring policy fallback."""

from __future__ import annotations

from iterative_context.graph_models import Graph, GraphNode
from iterative_context.scoring import score_degree


def score_fn(node: GraphNode, graph: Graph, step: int) -> float:
    """Deterministic local policy used when no injected scorer is provided."""
    # Degree-only scoring keeps behavior predictable while differing from the default.
    return score_degree(node, graph, step)


__all__ = ["score_fn"]
