def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The priority combines several signals while staying fully deterministic:

    * Base weight based on the node ``state``.
    * Structural signal – total degree (in + out) with a modest cap.
    * Token‑based signal for ``anchor`` and ``resolved`` nodes, encouraging
      token‑rich nodes.
    * Step‑based encouragement that grows with the traversal step.
    * A tiny deterministic tie‑breaker using the node identifier.
    """

    state = getattr(node, "state", None)
    total: float = 0.0

    # ---------------------------------------------------------------------
    # Base weight by node state.
    # ---------------------------------------------------------------------
    if state == "anchor":
        total += 14.0
    elif state == "pending":
        total += 6.0
    elif state == "resolved":
        total += 3.0
    elif state == "pruned":
        total -= 120.0

    # ---------------------------------------------------------------------
    # Structural signal – degree (capped for stability).
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 40.0) * 0.45

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes.
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_val = min(float(tokens), 200.0)
            total += token_val * 0.10

    # ---------------------------------------------------------------------
    # Step‑based encouragement.
    # ---------------------------------------------------------------------
    total += min(float(step), 8.0) * 0.18

    # ---------------------------------------------------------------------
    # Deterministic tie‑breaker using node identifier.
    # ---------------------------------------------------------------------
    tie = sum(ord(ch) for ch in str(node.id)) % 13
    total += tie * 0.001

    return float(total)