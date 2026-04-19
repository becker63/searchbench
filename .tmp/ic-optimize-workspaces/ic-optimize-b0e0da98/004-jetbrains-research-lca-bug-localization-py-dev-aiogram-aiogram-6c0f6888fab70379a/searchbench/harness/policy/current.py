def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The score combines:
    * A base weight based on the node's ``state``.
    * The node's degree (capped for stability).
    * Token information for ``anchor`` and ``resolved`` nodes.
    * A modest step‑based encouragement (capped).
    """

    # Base weight by state
    state = getattr(node, "state", None)
    base_weights = {
        "anchor": 12.0,
        "pending": 5.0,
        "resolved": 2.0,
        "pruned": -100.0,
    }
    total = base_weights.get(state, 0.0)

    # Structural signal – degree (capped for stability)
    try:
        degree = graph.degree(node.id)  # sum of in‑ and out‑degree
    except Exception:
        degree = 0
    total += min(float(degree), 30.0) * 0.4

    # Token‑based signal for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 150.0) * 0.07

    # Step‑based encouragement (capped at 5 steps)
    total += min(float(step), 5.0) * 0.15

    return float(total)