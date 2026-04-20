def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The score combines:
    * a base weight per node state,
    * the total degree (in + out) capped at 30 and scaled,
    * token count for anchor/resolved nodes,
    * a step‑based encouragement capped at 7.
    """

    # ---------------------------------------------------------------------
    # Base weight according to node state
    # ---------------------------------------------------------------------
    state = node.state
    base_weights = {
        "anchor": 15.0,
        "pending": 5.0,
        "resolved": 4.0,
        "pruned": -120.0,
    }
    total: float = base_weights.get(state, 0.0)

    # ---------------------------------------------------------------------
    # Structural degree signal (capped for stability)
    # ---------------------------------------------------------------------
    if node.id in graph:
        deg = graph.in_degree(node.id) + graph.out_degree(node.id)
    else:
        deg = 0
    total += min(float(deg), 30.0) * 0.5

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 150.0) * 0.12

    # ---------------------------------------------------------------------
    # Step‑based encouragement (capped at 7 steps)
    # ---------------------------------------------------------------------
    total += min(float(step), 7.0) * 0.15

    return float(total)