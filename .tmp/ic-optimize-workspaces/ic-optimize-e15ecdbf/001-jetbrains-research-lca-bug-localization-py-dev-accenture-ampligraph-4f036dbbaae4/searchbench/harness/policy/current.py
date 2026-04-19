def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a node during frontier expansion.

    The function combines a base weight derived from the node's ``state`` with
    structural and token‑based signals. All constants are chosen to favour
    ``anchor`` nodes (which tend to improve the *gold_hop* metric) and to reward
    token‑rich nodes (improving *token_efficiency*) while keeping the behaviour
    deterministic and simple.
    """
    # Base weight per state
    base_weights = {
        "anchor": 15.0,
        "pending": 9.0,
        "resolved": 6.0,
        "pruned": -50.0,
    }
    state = getattr(node, "state", None)
    total: float = base_weights.get(state, 0.0)

    # Structural signal: total degree (in + out). Cap to avoid runaway scores.
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        degree = 0
    total += min(float(degree), 30.0) * 0.7

    # Token signal for ``anchor`` and ``resolved`` nodes.
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 200.0) * 0.2

    # Step‑based encouragement – modest, capped.
    total += min(float(step), 20.0) * 0.1

    return float(total)