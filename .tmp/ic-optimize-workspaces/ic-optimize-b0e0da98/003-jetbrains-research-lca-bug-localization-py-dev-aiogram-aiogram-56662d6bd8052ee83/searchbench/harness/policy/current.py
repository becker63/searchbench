def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Compute a deterministic priority for *node*.

    The calculation combines several deterministic signals:

    * A base weight determined by the node ``state``.
    * The node degree (in‑ plus out‑degree) capped for stability.
    * Token count for ``anchor`` and ``resolved`` nodes.
    * A small encouragement that grows with the traversal step (capped).
    """

    # ---------------------------------------------------------------------
    # Base weight by node state
    # ---------------------------------------------------------------------
    state = getattr(node, "state", None)
    base_weights = {
        "anchor": 15.0,
        "pending": 4.0,
        "resolved": 3.0,
        "pruned": -200.0,
    }
    total: float = base_weights.get(state, 0.0)

    # ---------------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 40.0) * 0.5

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
    total += min(float(step), 10.0) * 0.1

    return float(total)