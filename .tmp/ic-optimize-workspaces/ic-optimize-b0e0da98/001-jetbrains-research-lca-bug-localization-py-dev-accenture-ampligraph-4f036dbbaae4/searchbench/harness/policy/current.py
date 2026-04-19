def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for selecting a node during graph traversal.

    Combines several signals:
    * Base weight based on the node's ``state``.
    * Structural weight derived from the node's degree (in‑plus‑out) with a cap.
    * Token weight for ``anchor`` and ``resolved`` nodes.
    * A step‑based encouragement that grows with the traversal step.
    All operations rely only on built‑in functionality.
    """

    # Base weight by node state
    state = getattr(node, "state", None)
    total: float = 0.0

    if state == "anchor":
        total += 15.0
    elif state == "pending":
        total += 6.0
    elif state == "resolved":
        total += 3.0
    elif state == "pruned":
        # Strongly discourage pruned nodes.
        total -= 100.0

    # Structural signal – degree (capped for stability)
    try:
        deg = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        deg = 0
    total += min(float(deg), 40.0) * 0.5

    # Token‑based signal for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 200.0) * 0.1

    # Step‑based encouragement (capped to avoid runaway growth)
    total += min(float(step), 10.0) * 0.2

    return float(total)