def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority function for frontier selection.

    The priority is a weighted sum of several deterministic signals:
    * A base weight derived from the node's ``state``.
    * A modest bonus proportional to the node's token count (if present).
    * A capped contribution from the node's total degree (in‑plus‑out).
    * An encouragement for earlier traversal steps (higher priority at low ``step``).

    The function returns a ``float`` and contains no side effects.
    """

    # ---------------------------------------------------------------------
    # 1. Base weight based on the node's state.
    # ---------------------------------------------------------------------
    state = getattr(node, "state", None)
    base_weights = {
        "anchor": 30.0,
        "resolved": 15.0,
        "pending": 5.0,
        "pruned": -100.0,
    }
    priority: float = base_weights.get(state, 0.0)

    # ---------------------------------------------------------------------
    # 2. Token‑based bonus (if the node carries a ``tokens`` field).
    # ---------------------------------------------------------------------
    tokens = getattr(node, "tokens", None)
    if isinstance(tokens, (int, float)):
        # Each token contributes a small positive amount.
        priority += tokens * 0.01

    # ---------------------------------------------------------------------
    # 3. Structural signal – total degree (in + out), capped for stability.
    # ---------------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    # Cap degree to avoid runaway scores on high‑degree hubs.
    priority += min(degree, 20) * 0.4

    # ---------------------------------------------------------------------
    # 4. Step‑based encouragement – favor earlier steps.
    # ---------------------------------------------------------------------
    # Earlier steps get higher priority; after step 10 the encouragement is zero.
    priority += (10 - min(step, 10)) * 0.2

    return float(priority)