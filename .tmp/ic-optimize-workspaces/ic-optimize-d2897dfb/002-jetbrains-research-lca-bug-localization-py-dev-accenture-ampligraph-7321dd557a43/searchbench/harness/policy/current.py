def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    Combines several signals while staying fully deterministic:

    * Base weight by node state.
    * Node degree (in + out) with a modest cap.
    * Token information for ``anchor`` and ``resolved`` nodes.
    * Step‑based encouragement that grows with the traversal step.
    * Tiny tie‑breaker derived from the node identifier to guarantee a stable order.
    """

    # ---------------------------------------------------------------------
    # Base weight by node state
    # ---------------------------------------------------------------------
    state = getattr(node, "state", None)
    base_weights: dict[str, float] = {
        "anchor": 15.0,
        "pending": 6.0,
        "resolved": 3.0,
        "pruned": -100.0,
    }
    total: float = base_weights.get(state, 0.0)

    # ---------------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 40.0) * 0.45

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 200.0) * 0.08

    # ---------------------------------------------------------------------
    # Step‑based encouragement (capped at 10 steps)
    # ---------------------------------------------------------------------
    total += min(float(step), 10.0) * 0.12

    # ---------------------------------------------------------------------
    # Tie‑breaker – deterministic small value from the node identifier
    # ---------------------------------------------------------------------
    tie_breaker = sum(ord(ch) for ch in str(node.id)) * 1e-5
    total += tie_breaker

    return float(total)