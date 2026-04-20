def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The calculation combines several signals while remaining fully deterministic:

    * A base weight depending on the node ``state``.
    * The node degree (sum of in‑ and out‑degree) limited to a cap for stability.
    * Token information for ``anchor`` and ``resolved`` nodes.
    * A step‑based encouragement that grows with the traversal step.
    * A tiny deterministic tie‑breaker based on the node identifier.
    """

    # ---------------------------------------------------------------------
    # Base weight by node state
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 30.0) * 0.4

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes (slightly higher weight)
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_val = min(float(tokens), 150.0)
            total += token_val * 0.1  # increased contribution

    # ---------------------------------------------------------------------
    # Step‑based encouragement (capped at 5 steps, higher weight)
    # ---------------------------------------------------------------------
    total += min(float(step), 5.0) * 0.2

    # ---------------------------------------------------------------------
    # Tiny deterministic tie‑breaker – based on the identifier string
    # ---------------------------------------------------------------------
    tie = (sum(ord(c) for c in str(node.id)) % 1000) * 0.0001
    total += tie

    return float(total)