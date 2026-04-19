def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    Combines lightweight signals while remaining fully deterministic:

    * **Base weight** – static boost based on the node's state.
    * **Structural signal** – total degree (in + out) of the node, capped.
    * **Token signal** – for anchor and resolved nodes, the ``tokens`` field is
      factored in with a gentle scaling factor, also capped.
    * **Step encouragement** – small bonus that grows with the traversal step
      number (up to a limit) to favour newer frontier nodes.
    """

    # Base weight according to node state
    state = getattr(node, "state", None)
    total: float = 0.0

    if state == "anchor":
        total += 15.0
    elif state == "pending":
        total += 6.0
    elif state == "resolved":
        total += 3.0
    elif state == "pruned":
        total -= 100.0

    # Structural signal – degree (capped for stability)
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 30.0) * 0.5

    # Token‑based signal for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_factor = 0.12 if state == "resolved" else 0.1
            total += min(float(tokens), 150.0) * token_factor

    # Step‑based encouragement – grows slowly and caps at step 10
    total += min(float(step), 10.0) * 0.1

    return float(total)