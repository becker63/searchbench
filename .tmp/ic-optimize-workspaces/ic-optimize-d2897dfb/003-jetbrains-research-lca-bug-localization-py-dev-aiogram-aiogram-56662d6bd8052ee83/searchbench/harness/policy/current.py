def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for graph traversal.

    The function combines several deterministic signals with slightly
    adjusted coefficients to favor gold (anchor) nodes, pending (issue) nodes,
    and token efficiency while preserving the original deterministic
    behavior.
    """
    state = getattr(node, "state", None)
    total: float = 0.0

    # Base weight by node state – tuned for better gold/issue promotion.
    if state == "anchor":
        total += 25.0  # increased credit for anchor (gold) nodes
    elif state == "pending":
        total += 8.0   # higher credit for pending (issue) nodes
    elif state == "resolved":
        total += 4.0
    elif state == "pruned":
        total -= 100.0  # milder penalty to keep pruning but less extreme

    # Structural signal – degree (capped for stability) with a slightly higher factor.
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 40.0) * 0.5

    # Token‑based signal for anchor and resolved nodes – increased influence.
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 200.0) * 0.12

    # Step‑based encouragement – modestly higher to keep forward progress.
    total += min(float(step), 8.0) * 0.15

    return float(total)