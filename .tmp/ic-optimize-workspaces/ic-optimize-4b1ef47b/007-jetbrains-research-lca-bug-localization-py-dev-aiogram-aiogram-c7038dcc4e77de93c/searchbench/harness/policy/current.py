def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The priority combines a small set of deterministic signals:

    * A base weight depending on the node ``state``.
    * The node degree (in+out) capped for stability.
    * Optional ``tokens`` value for anchor or resolved nodes.
    * A modest step‑based encouragement that grows slowly with the traversal step.
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
        total += 3.0
    elif state == "pruned":
        total -= 100.0

    # ---------------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 30.0) * 0.5

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_val = min(float(tokens), 100.0)
            total += token_val * 0.05

    # ---------------------------------------------------------------------
    # Step‑based encouragement (small, monotonic)
    # ---------------------------------------------------------------------
    total += min(float(step), 5.0) * 0.1

    return float(total)