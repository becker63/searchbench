def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The function assigns a numeric score that the traversal algorithm uses to
    decide which node to expand next. Larger scores are preferred.

    It combines several signals while remaining fully deterministic:

    * A base weight depending on the node ``state``.
    * Structural weight derived from the node's degree (capped for stability).
    * Token‑based signal for ``anchor`` and ``resolved`` nodes.
    * A modest step‑based encouragement that grows with the traversal step.
    * A tiny deterministic tie‑breaker based on the node identifier to avoid
      exact score collisions.
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
    total += min(float(degree), 30.0) * 0.6  # slightly higher weight for degree

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 150.0) * 0.2  # increased token weight

    # ---------------------------------------------------------------------
    # Step‑based encouragement (capped at 5 steps)
    # ---------------------------------------------------------------------
    total += min(float(step), 5.0) * 0.3  # slightly stronger step incentive

    # ---------------------------------------------------------------------
    # Tiny deterministic tie‑breaker based on node id
    # ---------------------------------------------------------------------
    # This adds a minuscule value derived from the first character of the id
    # ensuring that nodes with otherwise identical scores are ordered
    # consistently without affecting the overall ranking.
    if isinstance(node.id, str) and node.id:
        total += (ord(node.id[0]) % 10) * 1e-6

    return float(total)