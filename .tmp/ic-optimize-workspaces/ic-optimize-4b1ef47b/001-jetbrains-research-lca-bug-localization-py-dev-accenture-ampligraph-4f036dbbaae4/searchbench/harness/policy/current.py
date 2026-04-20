def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for frontier selection.

    Combines simple signals:
    * Base weight by node state.
    * Node degree (in + out), capped for stability.
    * Token count for anchor/resolved nodes, capped and scaled.
    * Step‑based encouragement, capped.
    """

    # Base weight according to node state
    state = getattr(node, "state", None)
    base_weights = {
        "anchor": 15.0,
        "pending": 5.0,
        "resolved": 3.0,
        "pruned": -100.0,
    }
    total: float = base_weights.get(state, 0.0)

    # Structural signal – total degree (capped for stability)
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 20.0) * 0.5

    # Token‑based signal for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 200.0) * 0.1

    # Step‑based encouragement (capped)
    total += min(float(step), 10.0) * 0.1

    return float(total)