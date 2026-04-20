def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a node during frontier selection.

    The priority combines several lightweight signals:

    * ``base`` – fixed weight based on the node's ``state``.
    * ``degree`` – total in‑ and out‑degree of the node (capped for stability).
    * ``tokens`` – token count for ``anchor`` and ``resolved`` nodes (capped).
    * ``step`` – encouragement that grows with the traversal step (capped).
    """

    # Base weight by node state
    state = getattr(node, "state", None)
    base_weights = {
        "anchor": 10.0,
        "pending": 5.0,
        "resolved": 3.0,
        "pruned": -50.0,
    }
    total: float = base_weights.get(state, 0.0)

    # Structural signal – degree (capped for stability)
    degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    total += min(degree, 40) * 0.3

    # Token‑based signal for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 200.0) * 0.12

    # Step‑based encouragement (capped for stability)
    total += min(step, 10) * 0.15

    return float(total)