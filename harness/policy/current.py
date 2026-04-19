def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The function combines several simple signals while remaining fully deterministic:

    * A base weight depending on the node ``state``.
    * The node degree (sum of in‑ and out‑degree) limited to a cap for stability.
    * Token information for ``anchor`` and ``resolved`` nodes.
    * A modest step‑based encouragement that grows with the traversal step.
    """

    # ---------------------------------------------------------------------
    # Base weight by node state
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 30.0) * 0.4

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_val = min(float(tokens), 150.0)
            total += token_val * 0.07

    # ---------------------------------------------------------------------
    # Step‑based encouragement (capped at 5 steps)
    # ---------------------------------------------------------------------
    total += min(float(step), 5.0) * 0.15

    return float(total)