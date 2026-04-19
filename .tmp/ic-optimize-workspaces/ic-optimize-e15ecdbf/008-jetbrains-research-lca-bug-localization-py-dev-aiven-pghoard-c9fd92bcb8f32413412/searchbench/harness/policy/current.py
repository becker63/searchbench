from __future__ import annotations

def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority function for graph traversal.

    The function combines several inexpensive, deterministic signals:

    * A base weight depending on the node ``state``.
    * Structural signal – total degree (capped for stability).
    * Token‑based signal for ``anchor`` and ``resolved`` nodes.
    * Step‑based encouragement that grows with the traversal step.

    All operations use only built‑in types and attributes present on the
    ``GraphNode`` and ``Graph`` objects, ensuring the function is pure and
    side‑effect free.
    """

    # ---------------------------------------------------------------------
    # Base weight by node state (higher priority for nodes we want to expand).
    # ---------------------------------------------------------------------
    state = getattr(node, "state", None)
    total: float = 0.0
    if state == "anchor":
        total += 15.0
    elif state == "pending":
        total += 6.0
    elif state == "resolved":
        total += 3.0
    elif state == "pruned":
        # Pruned nodes are never selected.
        total -= 1e6

    # ---------------------------------------------------------------------
    # Structural signal – total degree (in + out), capped for stability.
    # ---------------------------------------------------------------------
    try:
        deg = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        deg = 0
    total += min(float(deg), 40.0) * 0.5

    # ---------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes.
    # ---------------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 200.0) * 0.1

    # ---------------------------------------------------------------------
    # Step‑based encouragement – favors later steps (capped at 10).
    # ---------------------------------------------------------------------
    total += min(float(step), 10.0) * 0.2

    return float(total)