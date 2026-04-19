def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    Combines several signals while staying fully deterministic:
    * Base weight depending on the node ``state``.
    * Structural signal – degree (capped for stability).
    * Token‑based signal for ``anchor`` and ``resolved`` nodes.
    * Step‑based encouragement that grows with the traversal step.
    """

    # Base weight by node state
    state = getattr(node, "state", None)
    total: float = 0.0

    if state == "anchor":
        total += 15.0
    elif state == "pending":
        total += 6.0
    elif state == "resolved":
        total += 3.0
    elif state == "pruned":
        total -= 200.0

    # Structural signal – degree (capped for stability)
    degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    total += min(float(degree), 40.0) * 0.5

    # Token‑based signal for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 200.0) * 0.1

    # Step‑based encouragement (capped at 10 steps)
    total += min(float(step), 10.0) * 0.2

    return float(total)