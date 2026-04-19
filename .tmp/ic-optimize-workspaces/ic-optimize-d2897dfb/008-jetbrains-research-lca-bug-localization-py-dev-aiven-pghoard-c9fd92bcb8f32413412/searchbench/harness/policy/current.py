def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for selecting the next node to expand.

    The score favours *anchor* nodes, gives a modest boost to *pending* and
    *resolved* nodes, and heavily penalises *pruned* nodes. It also adds a small
    contribution based on the node's degree, token count (when applicable), and
    the current traversal step to encourage progress.
    """

    # Base weight by node state.
    state = getattr(node, "state", None)
    base_weights = {
        "anchor": 30.0,
        "pending": 10.0,
        "resolved": 5.0,
        "pruned": -1_000_000.0,
    }
    total: float = base_weights.get(state, 0.0)

    # Degree contribution (capped for stability).
    nid = getattr(node, "id", None)
    try:
        degree = graph.in_degree(nid) + graph.out_degree(nid)
    except Exception:
        degree = 0
    total += min(degree, 20) * 0.5

    # Token contribution for anchor and resolved nodes.
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 100.0) * 0.1

    # Step contribution (modest linear boost, capped).
    total += min(step, 10) * 0.2

    return float(total)