def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node used in expansion selection.

    The score combines:
    * a base value depending on the node's state,
    * the node's degree in the graph (capped for stability),
    * a token‑based boost for ``anchor`` and ``resolved`` nodes,
    * a small incentive that grows with the traversal step (capped).
    All operations are pure, use only built‑in types, and avoid external imports.
    """
    # ---------------------------------------------------------------------
    # Base contribution from the node's state
    # ---------------------------------------------------------------------
    state = getattr(node, "state", None)
    base_scores = {
        "anchor": 12.0,
        "pending": 8.0,
        "resolved": 2.0,
        "pruned": -100.0,
    }
    total = base_scores.get(state, 0.0)

    # ---------------------------------------------------------------------
    # Structural signal – degree (sum of in‑ and out‑edges), capped for stability
    # ---------------------------------------------------------------------
    if hasattr(graph, "__contains__") and node.id in graph:
        degree = graph.degree(node.id)
    else:
        degree = 0
    total += min(float(degree), 30.0) * 0.4

    # ---------------------------------------------------------------------
    # Token‑based boost for anchor and resolved nodes
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 150.0) * 0.07

    # ---------------------------------------------------------------------
    # Step‑based encouragement (capped at 5 steps)
    # ---------------------------------------------------------------------
    total += min(float(step), 5.0) * 0.2

    return float(total)