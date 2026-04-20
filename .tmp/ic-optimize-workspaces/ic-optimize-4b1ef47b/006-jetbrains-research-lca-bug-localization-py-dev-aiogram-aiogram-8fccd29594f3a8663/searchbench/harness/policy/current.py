def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for selecting a node during graph traversal.

    The calculation combines several signals while staying fully deterministic:

    * Base weight by the node ``state``.
    * Structural degree (in + out) with a modest cap.
    * Token count for ``anchor`` and ``resolved`` nodes.
    * Step‑based encouragement that grows with the traversal step.
    """
    # -----------------------------------------------------------------
    # Base weight by node state
    # -----------------------------------------------------------------
    state = getattr(node, "state", None)
    total: float = 0.0

    if state == "anchor":
        total += 20.0
    elif state == "pending":
        total += 8.0
    elif state == "resolved":
        total += 5.0
    elif state == "pruned":
        total -= 80.0

    # -----------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # -----------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 40.0) * 0.8

    # -----------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # -----------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_val = min(float(tokens), 200.0)
            total += token_val * 0.12

    # -----------------------------------------------------------------
    # Step‑based encouragement (capped at 10 steps)
    # -----------------------------------------------------------------
    total += min(float(step), 10.0) * 0.3

    return float(total)