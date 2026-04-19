def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The function assigns a base score based on the node's state, then
    adds a modest contribution from structural connectivity (degree),
    token count (for anchor and resolved nodes), and the current traversal
    step. All contributions are capped and use fixed multipliers to keep the
    result deterministic and bounded.
    """

    # Base weight determined by the node's state
    state = getattr(node, "state", None)
    total: float = 0.0

    if state == "anchor":
        total += 15.0
    elif state == "pending":
        total += 6.0
    elif state == "resolved":
        total += 3.0
    elif state == "pruned":
        # Pruned nodes should never be selected; give them a very low score.
        total -= 100.0

    # Structural signal: degree (in‑ + out‑degree), capped for stability.
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 40.0) * 0.5

    # Token‑based signal for anchor and resolved nodes.
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_val = min(float(tokens), 200.0)
            total += token_val * 0.08

    # Step‑based encouragement: grows with the traversal step, capped.
    total += min(float(step), 10.0) * 0.2

    return float(total)