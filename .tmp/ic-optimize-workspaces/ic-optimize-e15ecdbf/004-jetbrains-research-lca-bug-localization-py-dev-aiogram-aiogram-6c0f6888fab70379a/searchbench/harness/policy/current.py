def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a node within the expansion frontier.

    The score combines stable signals:
    * Base weight by node ``state``.
    * Node degree (in + out) capped for stability.
    * Token count for ``anchor`` and ``resolved`` nodes.
    * A modest step‑based encouragement that grows with the traversal step.
    """

    # Base weight by node state
    state = getattr(node, "state", None)
    total: float = 0.0

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
    total += min(float(degree), 30.0) * 0.45

    # Token‑based signal for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 150.0) * 0.1

    # Step‑based encouragement (capped at 5 steps)
    total += min(float(step), 5.0) * 0.2

    return float(total)