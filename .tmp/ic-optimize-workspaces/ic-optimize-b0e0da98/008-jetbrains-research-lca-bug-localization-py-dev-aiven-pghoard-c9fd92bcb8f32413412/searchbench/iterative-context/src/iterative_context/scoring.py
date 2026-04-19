# iterative-context/src/iterative_context/scoring.py

from __future__ import annotations

import hashlib

from iterative_context.graph_models import Graph, GraphNode


def score_v1(node: GraphNode, graph: Graph, step: int) -> float:
    """Baseline scoring function."""
    score = 0.0

    if node.state == "pending":
        score += 5.0
    if node.state == "anchor":
        score += 10.0

    degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    score += degree * 0.5

    tokens = getattr(node, "tokens", None)
    if tokens is not None:
        score += min(tokens, 50) * 0.1

    return score


def score_degree(node: GraphNode, graph: Graph, step: int) -> float:
    """Degree-only baseline."""
    return graph.in_degree(node.id) + graph.out_degree(node.id)


def score_random(node: GraphNode, graph: Graph, step: int) -> float:
    """Deterministic pseudo-random baseline."""
    key = f"{node.id}-{step}".encode()
    return int(hashlib.md5(key).hexdigest(), 16) % 1000


def default_score_fn(node: GraphNode, graph: Graph, step: int) -> float:
    """Default scoring entrypoint."""
    return score_v1(node, graph, step)


__all__ = ["score_v1", "score_degree", "score_random", "default_score_fn"]
