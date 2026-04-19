def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for graph traversal frontier.

    The score combines several lightweight signals:
    1. Base weight by node ``state``.
    2. Structural signal – total degree (capped for stability).
    3. Token signal for ``anchor`` and ``resolved`` nodes.
    4. Linear encouragement that grows with the traversal step.
    """
    # 1. Base weight by node state
    state = getattr(node, "state", None)
    total: float = 0.0

    if state == "pending":
        total += 8.0
    elif state == "anchor":
        total += 10.0
    elif state == "resolved":
        total += 2.0
    elif state == "pruned":
        total -= 100.0

    # 2. Structural signal – degree (capped for stability)
    degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    total += min(float(degree), 20.0) * 0.5

    # 3. Token‑based signal for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 150.0) * 0.07

    # 4. Step‑based encouragement – linear growth
    total += float(step) * 0.2

    return float(total)