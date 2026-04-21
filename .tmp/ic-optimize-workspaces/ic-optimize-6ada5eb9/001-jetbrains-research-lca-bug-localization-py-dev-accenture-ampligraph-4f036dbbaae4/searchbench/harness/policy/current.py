def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority score for a graph node.

    The score combines:
    * a base value depending on the node's ``state``;
    * a contribution from the node's total degree (in‑ + out‑degree), capped for stability;
    * a token‑based boost for ``anchor`` and ``resolved`` nodes, also capped;
    * a small encouragement for early traversal steps.

    The function never raises and always returns a ``float``.
    """

    # Base values per state – higher is more desirable.
    base_scores = {
        "anchor": 15.0,
        "pending": 6.0,
        "resolved": 3.0,
        "pruned": -100.0,
    }
    state = getattr(node, "state", None)
    base = base_scores.get(state, 0.0)

    # Structural contribution: total degree (in + out), capped to avoid runaway scores.
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    deg_score = min(float(degree), 40.0) * 0.35

    # Token contribution for anchor / resolved nodes.
    token_score = 0.0
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", None)
        if isinstance(tokens, (int, float)):
            token_score = min(float(tokens), 200.0) * 0.08

    # Small encouragement for early steps (first ten steps).
    step_score = min(float(step), 10.0) * 0.1

    return float(base + deg_score + token_score + step_score)