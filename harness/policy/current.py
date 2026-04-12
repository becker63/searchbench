from __future__ import annotations

from iterative_context.graph_models import Graph, GraphNode  # type: ignore[import-not-found]


def _degree(node: GraphNode, graph: Graph) -> int:
    return int(graph.in_degree(node.id) + graph.out_degree(node.id))


def _bounded_tokens(node: GraphNode, cap: float = 200.0) -> float:
    if node.state == "anchor":
        tokens_val = node.tokens
        if isinstance(tokens_val, (int, float)):
            return min(float(tokens_val), cap)
        return 0.0
    if node.state == "resolved":
        return min(float(node.tokens), cap)
    return 0.0


def frontier_priority(node: GraphNode, graph: Graph, step: int) -> float:
    """
    Deterministic traversal priority aligned to SelectionCallable.

    Priorities:
      - anchors first, then pending (unexplored) nodes
      - resolved nodes get a small positive weight (still useful context)
      - pruned nodes are strongly demoted
      - local graph degree adds structural signal
      - token content on anchor/resolved provides evidence signal
      - step provides mild lookahead encouragement later in traversal
    """
    total = 0.0

    if node.state == "anchor":
        total += 10.0
    elif node.state == "pending":
        total += 5.0
    elif node.state == "resolved":
        total += 1.5
    elif node.state == "pruned":
        total -= 100.0

    total += min(float(_degree(node, graph)), 25.0) * 0.5

    tokens = _bounded_tokens(node, cap=120.0)
    if tokens > 0.0:
        total += tokens * 0.08

    total += min(float(step), 5.0) * 0.2

    return float(total)
