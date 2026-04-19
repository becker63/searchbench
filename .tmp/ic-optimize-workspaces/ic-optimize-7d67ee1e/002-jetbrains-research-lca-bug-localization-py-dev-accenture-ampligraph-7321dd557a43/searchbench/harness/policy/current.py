def frontier_priority(node: GraphNode, graph: Graph, step: int) -> float:
    """Deterministic priority for a graph node.

    The function combines several stable signals:
    * Base weight based on the node's ``state``.
    * Structural degree (in‑plus‑out) with a modest cap.
    * Token count for ``anchor`` and ``resolved`` nodes.
    * Step‑based encouragement that grows early in the traversal.
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
    total += min(float(degree), 30.0) * 0.6  # slightly higher weight than baseline

    # Token‑based signal for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 150.0) * 0.14  # increased contribution

    # Step‑based encouragement (capped at 5 steps)
    total += min(float(step), 5.0) * 0.25

    return float(total)