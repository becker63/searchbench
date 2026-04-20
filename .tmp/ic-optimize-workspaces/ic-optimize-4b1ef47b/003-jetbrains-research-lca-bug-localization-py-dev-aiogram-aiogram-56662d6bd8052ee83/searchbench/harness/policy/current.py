def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The scoring combines several signals while staying fully deterministic:

    * State‑based base weight – favoring anchor and pending nodes.
    * Structural signal – node degree (capped for stability).
    * Token‑based signal for ``anchor`` and ``resolved`` nodes.
    * Step‑based encouragement – grows with traversal depth but caps early.
    """

    # ---------------------------------------------------------------------
    # Base weight by node state
    # ---------------------------------------------------------------------
    state = getattr(node, "state", None)
    total: float = 0.0

    if state == "anchor":
        total += 20.0
    elif state == "pending":
        total += 8.0
    elif state == "resolved":
        total += 5.0
    elif state == "pruned":
        # Strongly discourage pruned nodes
        total -= 200.0

    # ---------------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 50.0) * 0.6

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 200.0) * 0.15

    # ---------------------------------------------------------------------
    # Step‑based encouragement (capped at 10 steps)
    # ---------------------------------------------------------------------
    total += min(float(step), 10.0) * 0.25

    return float(total)