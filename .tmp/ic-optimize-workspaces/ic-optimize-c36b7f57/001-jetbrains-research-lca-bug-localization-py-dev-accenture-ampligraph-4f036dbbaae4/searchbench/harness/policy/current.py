def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The function combines inexpensive, reproducible signals:

    * A base weight derived from the node ``state``.
    * The (in + out) degree of the node, capped for stability.
    * Token count for ``anchor`` and ``resolved`` nodes.
    * A modest step‑based encouragement that grows with the traversal step.
    """

    # ---------------------------------------------------------------------
    # Base weight by node state
    # ---------------------------------------------------------------------
    base_weights = {
        "anchor": 10.0,
        "pending": 6.0,
        "resolved": 3.0,
        "pruned": -50.0,
    }
    state = getattr(node, "state", None)
    priority: float = base_weights.get(state, 0.0)

    # ---------------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    if degree > 20:
        degree = 20
    priority += degree * 0.5

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            t = float(tokens)
            if t > 100.0:
                t = 100.0
            priority += t * 0.08

    # ---------------------------------------------------------------------
    # Step‑based encouragement (capped at 10 steps)
    # ---------------------------------------------------------------------
    step_bonus = step if step < 10 else 10
    priority += step_bonus * 0.1

    return float(priority)