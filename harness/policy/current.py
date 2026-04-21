from __future__ import annotations

from iterative_context.graph_models import Graph, GraphNode
from iterative_context.types import SelectionCallable


def _degree(node: GraphNode, graph: Graph) -> int:
    """Return the total degree (in + out) of *node*.

    The graph provides ``in_degree`` and ``out_degree`` methods that accept a node identifier.
    The sum is cast to ``int`` for safety.
    """
    return int(graph.in_degree(node.id) + graph.out_degree(node.id))


def _bounded_tokens(node: GraphNode, cap: float = 200.0) -> float:
    """Return the token count for *node*, bounded by *cap*.

    Only ``anchor`` and ``resolved`` nodes expose a ``tokens`` attribute. Non‑numeric values are treated as ``0.0``.
    """
    if node.state == "anchor" or node.state == "resolved":
        t = getattr(node, "tokens", None)
        if isinstance(t, (int, float)):
            return min(float(t), cap)
        return 0.0
    return 0.0


def frontier_priority(node: GraphNode, graph: Graph, step: int) -> float:
    """Deterministic priority for frontier selection.

    The score combines:
    * a base weight depending on the node ``state``;
    * a degree contribution (capped) to favour well‑connected nodes;
    * a token contribution with a higher weight to improve token efficiency;
    * a modest step‑based boost to prefer newer frontier nodes.
    """
    score = 0.0

    # Base weight per state
    if node.state == "anchor":
        score += 12.0
    elif node.state == "pending":
        score += 5.0
    elif node.state == "resolved":
        score += 2.0
    elif node.state == "pruned":
        score -= 100.0

    # Degree contribution (capped at 30) – weight reduced to allocate more to tokens
    score += min(float(_degree(node, graph)), 30.0) * 0.2

    # Token contribution – increased weight for better token efficiency
    score += _bounded_tokens(node, cap=200.0) * 0.6

    # Small step‑based boost (capped at 5)
    score += min(float(step), 5.0) * 0.2

    return float(score)


priority_fn: SelectionCallable = frontier_priority