def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a graph node.

    The priority aggregates several signals with tuned weights to encourage
    exploration of promising regions while remaining deterministic.

    * **Base weight** – strong preference for ``anchor`` nodes, then ``pending``
      and ``resolved``. ``pruned`` nodes keep a large negative bias.
    * **Degree bonus** – sum of in‑ and out‑degree (capped) encourages moving
      through well‑connected areas.
    * **Token bonus** – for ``anchor`` and ``resolved`` nodes, a portion of the
      ``tokens`` count is added, favoring token‑rich locations.
    * **Step bonus** – modest increase that grows with the traversal step to
      keep the search progressing.
    """

    # Base weight by node state
    state = getattr(node, "state", None)
    base_weights = {
        "anchor": 40.0,
        "pending": 15.0,
        "resolved": 12.0,
        "pruned": -100.0,
    }
    total: float = base_weights.get(state, 0.0)

    # Degree contribution (capped for stability)
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 30.0) * 0.6

    # Token contribution for anchor and resolved nodes
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 1000.0) * 0.07

    # Step contribution (capped to avoid runaway growth)
    total += min(float(step), 15.0) * 0.12

    return float(total)