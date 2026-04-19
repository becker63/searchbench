def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    Combines a base weight based on ``state``, a capped degree contribution,
    optional token information for ``anchor`` and ``resolved`` nodes, and a modest
    step‑based encouragement.
    """

    # Base weight by node state
    state = getattr(node, "state", None)
    base_weights = {
        "anchor": 12.0,
        "pending": 5.0,
        "resolved": 2.0,
        "pruned": -100.0,
    }
    base = base_weights.get(state, 0.0)

    # Structural signal – degree (capped for stability)
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)  # type: ignore[attr-defined]
    except Exception:
        degree = 0
    degree_score = min(float(degree), 30.0) * 0.4

    # Token‑based signal for anchor and resolved nodes
    token_score = 0.0
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_score = min(float(tokens), 150.0) * 0.07

    # Step‑based encouragement (capped at 5 steps)
    step_score = min(float(step), 5.0) * 0.15

    return float(base + degree_score + token_score + step_score)