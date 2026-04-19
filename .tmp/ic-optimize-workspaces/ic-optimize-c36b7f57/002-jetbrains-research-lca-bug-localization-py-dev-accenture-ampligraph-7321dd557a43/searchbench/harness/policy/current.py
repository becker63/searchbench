def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The calculation combines a base weight by node state, a capped degree
    contribution, optional token information for anchor/resolved nodes, and a
    modest step‑based encouragement.  All operations are pure Python and rely
    only on the attributes defined in ``graph_models.GraphNode`` and the
    ``networkx`` DiGraph interface.
    """

    # ---------------------------------------------------------------------
    # Base weight derived from the node's ``state``.
    # ---------------------------------------------------------------------
    state = getattr(node, "state", None)
    total: float = 0.0
    if state == "anchor":
        total += 14.0
    elif state == "pending":
        total += 6.0
    elif state == "resolved":
        total += 3.0
    elif state == "pruned":
        total -= 100.0

    # ---------------------------------------------------------------------
    # Structural signal – sum of in‑ and out‑degree, capped for stability.
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 30.0) * 0.5

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes.
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 150.0) * 0.1

    # ---------------------------------------------------------------------
    # Step‑based encouragement – modest growth with a capped contribution.
    # ---------------------------------------------------------------------
    total += min(float(step), 5.0) * 0.2

    return float(total)