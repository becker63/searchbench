def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The function assigns a strong boost to *pending* nodes, moderate scores to
    ``anchor`` and ``resolved`` nodes (including token evidence), heavily
    penalises ``pruned`` nodes, and adds a step‑based encouragement. All
    operations use only built‑in functionality and the provided ``Graph`` and
    ``GraphNode`` models.
    """

    # Base weight according to node state.
    state = getattr(node, "state", None)
    base_weights = {
        "pending": 35.0,   # stronger boost for pending nodes
        "anchor": 6.0,
        "resolved": 9.0,
        "pruned": -200.0,
    }
    total: float = base_weights.get(state, 0.0)

    # Structural signal – combined in/out degree, capped for stability.
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)  # type: ignore[attr-defined]
    except Exception:
        degree = 0
    total += min(float(degree), 30.0) * 0.6

    # Token‑based signal – reward nodes that carry token evidence.
    tokens = getattr(node, "tokens", 0) or 0
    if isinstance(tokens, (int, float)):
        token_val = min(float(tokens), 300.0)
        if state in ("pending", "anchor", "resolved"):
            # Slightly higher scaling to improve token efficiency.
            total += token_val * 0.14

    # Step‑based encouragement – gently increase priority as traversal
    # progresses, encouraging deeper exploration later in the search.
    total += min(float(step), 20.0) * 0.30

    return float(total)