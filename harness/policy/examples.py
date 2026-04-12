from __future__ import annotations

"""
Example frontier-priority policies aligned to Iterative Context's traversal contract.

This file is meant to teach policy authors and the writer model the expected
shape of a traversal-priority function:

    frontier_priority(node: GraphNode, graph: Graph, step: int) -> float

It intentionally contains multiple examples with different styles:
- `frontier_priority_basic`: minimal and readable
- `frontier_priority_connectivity_first`: favors graph structure
- `frontier_priority_token_aware`: uses state narrowing and token information
- `frontier_priority_conservative`: heavily avoids pruned/resolved nodes
- `frontier_priority`: default balanced example

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


def frontier_priority_basic(node: GraphNode, graph: Graph, step: int) -> float:
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


def frontier_priority_connectivity_first(node: GraphNode, graph: Graph, step: int) -> float:
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


def frontier_priority_token_aware(node: GraphNode, graph: Graph, step: int) -> float:
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

    tokens = _bounded_tokens(node, cap=150.0)
    total += tokens * 0.07

    if tokens > 120.0:
        total -= (tokens - 120.0) * 0.02

    total += _degree(node, graph) * 0.35
    total += min(float(step), 6.0) * 0.2

    return float(total)


def frontier_priority_conservative(node: GraphNode, graph: Graph, step: int) -> float:
    """
    Conservative example:
    - heavily penalizes pruned nodes
    - modest penalty for resolved nodes
    - moderate encouragement for anchors
    """
    total = 0.0
    if node.state == "pruned":
        total -= 100.0
    elif node.state == "resolved":
        total -= 2.5
    elif node.state == "pending":
        total += 5.5
    elif node.state == "anchor":
        total += 10.5

    degree = _degree(node, graph)
    total += min(float(degree), 25.0) * 0.45

    tokens = _bounded_tokens(node, cap=120.0)
    if tokens > 0.0:
        total += tokens * 0.08

    total += min(float(step), 5.0) * 0.2

    return float(total)


def frontier_priority_balanced(node: GraphNode, graph: Graph, step: int) -> float:
    """
    Balanced example combining several traversal signals:
    - anchors/pending favored
    - resolved slightly positive
    - pruned heavily negative
    - degree weighted moderately
    - tokens on anchor/resolved contribute
    - iteration step provides light lookahead preference
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


def frontier_priority(node: GraphNode, graph: Graph, step: int) -> float:
    """
    Default frontier priority aligned to SelectionCallable.

    Searchbench can use this as the primary example while still exposing the
    other examples in this file for the writer to learn from.
    """
    return frontier_priority_balanced(node, graph, step)


__all__ = [
    "frontier_priority",
    "frontier_priority_basic",
    "frontier_priority_connectivity_first",
    "frontier_priority_token_aware",
    "frontier_priority_conservative",
    "frontier_priority_balanced",
]
