# Placeholder type definitions to satisfy static analysis without imports
class Graph:  # pragma: no cover
    pass

class GraphNode:  # pragma: no cover
    pass

def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The calculation combines several signals while staying fully deterministic:

    * Base weight depending on the node ``state``.
    * Structural degree (in‑ + out‑degree) with a modest cap.
    * Token count for ``anchor`` and ``resolved`` nodes.
    * Length of the evidence snippet (if present).
    * Step‑based encouragement that grows with the traversal step.
    """

    # ---------------------------------------------------------------------
    # Base weight by node state
    # ---------------------------------------------------------------------
    state = getattr(node, "state", None)
    total: float = 0.0

    if state == "anchor":
        total += 15.0
    elif state == "pending":
        total += 7.0
    elif state == "resolved":
        total += 4.0
    elif state == "pruned":
        total -= 100.0

    # ---------------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(degree, 30) * 0.6

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 300.0) * 0.15

    # ---------------------------------------------------------------------
    # Evidence snippet length signal (if present)
    # ---------------------------------------------------------------------
    evidence = getattr(node, "evidence", None)
    if evidence is not None:
        snippet = getattr(evidence, "snippet", "")
        if isinstance(snippet, str):
            total += min(len(snippet), 800) * 0.004

    # ---------------------------------------------------------------------
    # Step‑based encouragement (capped for stability)
    # ---------------------------------------------------------------------
    # Slightly increased weight to favour deeper exploration.
    total += min(step, 20) * 0.05

    return float(total)