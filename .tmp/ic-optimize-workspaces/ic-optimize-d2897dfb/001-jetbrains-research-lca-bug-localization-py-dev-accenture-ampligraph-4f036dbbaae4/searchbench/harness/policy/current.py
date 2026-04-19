def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Compute a deterministic priority for a node.

    Combines a state bias, degree information, token count (for anchor/resolved)
    and the current step index.
    """
    state = getattr(node, "state", None)
    total = 0.0
    if state == "anchor":
        total += 20.0
    elif state == "pending":
        total += 8.0
    elif state == "resolved":
        total += 4.0
    else:
        total -= 100.0

    try:
        deg = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        deg = 0
    total += min(float(deg), 60.0) * 0.6

    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 300.0) * 0.12

    total += min(float(step), 15.0) * 0.25
    return float(total)