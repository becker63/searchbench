def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    Updated weighting to favor nodes that are more promising for the search
    objectives while keeping the function pure and deterministic.

    The calculation consists of:
    * A base weight depending on the node ``state``.
    * Structural signal – the total degree (in + out) with a modest cap.
    * Token‑based signal for ``anchor`` and ``resolved`` nodes.
    * Step‑based encouragement that grows with the traversal step.
    """

    # ---------------------------------------------------------------------
    # Base weight by node state (slightly increased to reward useful states)
    # ---------------------------------------------------------------------
    state = getattr(node, "state", None)
    total: float = 0.0

    if state == "anchor":
        total += 15.0  # higher priority for anchor nodes
    elif state == "pending":
        total += 6.0   # modest boost for pending nodes
    elif state == "resolved":
        total += 3.0   # resolved nodes are valuable
    elif state == "pruned":
        total -= 100.0  # strongly discourage pruned nodes

    # ---------------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    # Cap the degree contribution to avoid runaway values
    total += min(float(degree), 40.0) * 0.5

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_val = min(float(tokens), 200.0)
            total += token_val * 0.10

    # ---------------------------------------------------------------------
    # Step‑based encouragement (allow a larger window)
    # ---------------------------------------------------------------------
    total += min(float(step), 10.0) * 0.20

    return float(total)