def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Calculate a deterministic priority for a graph node.

    The priority combines four simple signals:
    * a base weight derived from the node's ``state``;
    * a structural contribution based on the node's degree (in‑ + out‑degree);
    * a token‑based contribution for ``anchor`` and ``resolved`` nodes;
    * a modest encouragement that grows with the traversal step.
    All components are capped to keep the score stable.
    """

    # --- Base weight by node state ------------------------------------------------
    state = node.state
    if state == "anchor":
        base = 12.0
    elif state == "pending":
        base = 5.0
    elif state == "resolved":
        base = 2.0
    else:  # "pruned" or any unexpected state
        base = -100.0

    # --- Structural signal – total degree (capped) ------------------------------
    degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    degree_score = min(degree, 30) * 0.4

    # --- Token‑based signal for anchor / resolved nodes --------------------------
    token_score = 0.0
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_score = min(float(tokens), 150.0) * 0.07

    # --- Step‑based encouragement (capped at 5 steps) ----------------------------
    step_score = min(step, 5) * 0.15

    total = base + degree_score + token_score + step_score
    return float(total)