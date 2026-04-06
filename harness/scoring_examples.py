from __future__ import annotations

"""
Example scoring policies aligned to Iterative Context's public scorer contract.

This file is meant to teach policy authors and the writer model the expected
shape of a scoring function:

    score(node: GraphNode, graph: Graph, step: int) -> float

It intentionally contains multiple examples with different styles:
- `score_basic`: minimal and readable
- `score_connectivity_first`: favors graph structure
- `score_token_aware`: uses state narrowing and token information
- `score_conservative`: heavily avoids pruned/resolved nodes
- `score`: default balanced example

Guidelines demonstrated here:
- use direct typed access for stable fields like `node.id` and `node.state`
- use explicit state narrowing for variant-specific fields like `tokens`
- use graph methods directly when they are part of the intended contract
- keep policies deterministic and side-effect free
"""

from iterative_context.graph_models import (  # type: ignore[import-not-found]
    Graph,
    GraphNode,
)


def _degree(node: GraphNode, graph: Graph) -> int:
    """Return total local degree for a node."""
    return int(graph.in_degree(node.id) + graph.out_degree(node.id))


def _bounded_tokens(node: GraphNode, cap: float = 200.0) -> float:
    """
    Read token information only for node variants that actually carry it.
    Demonstrates explicit state narrowing instead of defensive getattr usage.
    """
    if node.state == "anchor":
        tokens_val = node.tokens
        if isinstance(tokens_val, (int, float)):
            return min(float(tokens_val), cap)
        return 0.0

    if node.state == "resolved":
        return min(float(node.tokens), cap)

    return 0.0


def score_basic(node: GraphNode, graph: Graph, step: int) -> float:
    """
    Minimal example:
    - prefer pending and anchor nodes
    - reward local degree
    - lightly reward tokens on anchor/resolved nodes
    """
    total = 0.0

    if node.state == "pending":
        total += 5.0
    elif node.state == "anchor":
        total += 10.0
    elif node.state == "resolved":
        total += 1.0
    elif node.state == "pruned":
        total -= 100.0

    total += _degree(node, graph) * 0.5
    total += _bounded_tokens(node, cap=50.0) * 0.1
    total += min(float(step), 5.0) * 0.25

    return float(total)


def score_connectivity_first(node: GraphNode, graph: Graph, step: int) -> float:
    """
    Connectivity-heavy example:
    - still respects state
    - cares much more about graph structure than tokens
    """
    total = 0.0
    degree = _degree(node, graph)

    if node.state == "anchor":
        total += 8.0
    elif node.state == "pending":
        total += 4.0
    elif node.state == "resolved":
        total -= 1.0
    elif node.state == "pruned":
        total -= 100.0

    total += min(float(degree), 30.0) * 0.9

    # Small iteration bonus to encourage lookahead paths later in traversal.
    total += min(float(step), 8.0) * 0.15

    return float(total)


def score_token_aware(node: GraphNode, graph: Graph, step: int) -> float:
    """
    Token-aware example:
    - prefers anchors
    - uses token information for anchor/resolved nodes
    - gently penalizes very large contexts
    """
    total = 0.0

    if node.state == "anchor":
        total += 10.0
    elif node.state == "pending":
        total += 6.0
    elif node.state == "resolved":
        total += 2.0
    elif node.state == "pruned":
        total -= 100.0

    degree = _degree(node, graph)
    total += min(float(degree), 20.0) * 0.35

    tokens = _bounded_tokens(node, cap=500.0)
    if tokens > 0.0:
        # Reward presence of useful evidence/tokens, but taper off.
        total += min(tokens, 120.0) * 0.06
        # Slight penalty for very large token footprints.
        if tokens > 150.0:
            total -= (tokens - 150.0) * 0.015

    total += min(float(step), 6.0) * 0.2

    return float(total)


def score_conservative(node: GraphNode, graph: Graph, step: int) -> float:
    """
    Conservative example:
    - strongly prefers pending/anchor
    - strongly demotes resolved/pruned
    - uses degree as a secondary signal
    """
    total = 0.0

    if node.state == "pending":
        total += 8.0
    elif node.state == "anchor":
        total += 12.0
    elif node.state == "resolved":
        total -= 6.0
    elif node.state == "pruned":
        total -= 100.0

    degree = _degree(node, graph)
    total += min(float(degree), 15.0) * 0.3

    if node.state == "anchor":
        tokens = _bounded_tokens(node, cap=100.0)
        total += tokens * 0.04

    # Conservative policies often care less about step.
    total += min(float(step), 3.0) * 0.1

    return float(total)


def score_balanced(node: GraphNode, graph: Graph, step: int) -> float:
    """
    Balanced example:
    - combines state priority
    - moderate structural reward
    - moderate token reward
    - mild step awareness
    """
    total = 0.0

    if node.state == "pending":
        total += 5.5
    elif node.state == "anchor":
        total += 10.5
    elif node.state == "resolved":
        total += 1.5
    elif node.state == "pruned":
        total -= 100.0

    degree = _degree(node, graph)
    total += min(float(degree), 25.0) * 0.45

    tokens = _bounded_tokens(node, cap=120.0)
    if tokens > 0.0:
        total += tokens * 0.08

    total += min(float(step), 5.0) * 0.2

    return float(total)


def score(node: GraphNode, graph: Graph, step: int) -> float:
    """
    Default example scoring function aligned to SelectionCallable.

    Searchbench can use this as the primary example while still exposing the
    other examples in this file for the writer to learn from.
    """
    return score_balanced(node, graph, step)


__all__ = [
    "score",
    "score_basic",
    "score_connectivity_first",
    "score_token_aware",
    "score_conservative",
    "score_balanced",
]
