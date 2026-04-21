def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The function combines several signals in a lightweight, deterministic way:

    * Base weight based on the node's ``state``.
    * A bias toward nodes with a smaller total degree (in‑ + out‑degree).
    * A token reward for ``anchor`` and ``resolved`` nodes.
    * A modest step‑based encouragement that grows with the traversal step.
    """

    # ---------------------------------------------------------------------
    # Base weight – primary influence.
    # ---------------------------------------------------------------------
    state = getattr(node, "state", None)
    total: float = 0.0
    if state == "anchor":
        total += 16.0  # favour anchor nodes a bit more
    elif state == "pending":
        total += 12.0
    elif state == "resolved":
        total += 11.0
    elif state == "pruned":
        total -= 80.0  # discourage pruned nodes

    # ---------------------------------------------------------------------
    # Structural signal – favour nodes with lower degree (in + out).
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)  # type: ignore[arg-type]
    except Exception:
        degree = 0
    # Cap the degree to keep the contribution bounded.
    capped = degree if degree < 30 else 30
    # Inverse degree influence: lower degree => higher boost.
    total += (30 - capped) * 1.0

    # ---------------------------------------------------------------------
    # Token‑based signal – only for anchor and resolved nodes.
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        token_val = getattr(node, "tokens", 0) or 0
        if isinstance(token_val, (int, float)):
            # Limit token contribution to avoid runaway values.
            token_val = token_val if token_val < 1000 else 1000
            total += token_val * 0.9

    # ---------------------------------------------------------------------
    # Step‑based encouragement – small linear boost that grows with step.
    # ---------------------------------------------------------------------
    total += min(step, 10) * 0.05

    return float(total)