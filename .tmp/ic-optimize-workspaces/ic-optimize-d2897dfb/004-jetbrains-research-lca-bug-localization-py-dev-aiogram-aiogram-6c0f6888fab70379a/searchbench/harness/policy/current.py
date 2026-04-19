def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    Combines signals from node state, degree, token count and traversal step.
    Returns a float score used by the frontier selection algorithm.
    """
    # Base weight by node state
    state = getattr(node, "state", None)
    total: float = 0.0
    if state == "anchor":
        total += 12.0
    elif state == "pending":
        total += 5.0
    elif state == "resolved":
        total += 2.0
    elif state == "pruned":
        total -= 100.0

    # Structural weight – node degree (capped for stability)
    try:
        deg = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        deg = 0
    total += min(float(deg), 30.0) * 0.5

    # Token weight for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 150.0) * 0.10

    # Step weight – encourages deeper traversal as steps increase
    total += min(float(step), 5.0) * 0.20

    return float(total)