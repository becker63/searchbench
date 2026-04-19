def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The score combines a base weight based on the node's state, a capped
    degree contribution, optional token information, and a modest step‑based
    encouragement.
    """
    state = getattr(node, "state", None)
    total: float = 0.0

    # Base weight by node state
    if state == "anchor":
        total += 12.0
    elif state == "pending":
        total += 5.0
    elif state == "resolved":
        total += 2.0
    elif state == "pruned":
        total -= 100.0

    # Structural signal – degree (capped for stability)
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 30.0) * 0.4

    # Token‑based signal for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 150.0) * 0.07

    # Step‑based encouragement (capped at 5 steps)
    total += min(float(step), 5.0) * 0.15

    return float(total)