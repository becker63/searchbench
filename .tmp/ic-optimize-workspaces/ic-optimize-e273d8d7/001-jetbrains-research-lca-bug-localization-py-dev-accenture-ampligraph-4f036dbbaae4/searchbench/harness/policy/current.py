def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a frontier node.

    The score combines several weighted components:
    * ``base`` – weight based on the node's ``state``.
    * ``degree_score`` – contribution from the node's degree (capped).
    * ``token_score`` – boost for ``anchor`` and ``resolved`` nodes.
    * ``step_score`` – modest bonus that grows with the traversal step.
    * ``tie`` – tiny deterministic tie‑breaker derived from the identifier.
    """

    # Base weight by state (anchor > resolved > pending > others)
    state = getattr(node, "state", "")
    base_weights = {"anchor": 40.0, "resolved": 30.0, "pending": 10.0}
    base = base_weights.get(state, -10.0)

    # Degree contribution (in + out), capped at 30 for stability
    try:
        deg = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        deg = 0
    if deg > 30:
        deg = 30
    degree_score = deg * 0.8

    # Token boost – only for anchor or resolved nodes
    token_score = 0.0
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        try:
            t = float(tokens)
        except Exception:
            t = 0.0
        if t > 300.0:
            t = 300.0
        token_score = t * 0.7

    # Step bonus – linear growth, capped at 20 steps
    step_bonus = step if step < 20 else 20
    step_score = step_bonus * 0.5

    # Deterministic tie‑breaker based on the identifier
    tie = 0.0
    if isinstance(node.id, str):
        tie = (sum(ord(c) for c in node.id) % 5) * 0.01

    return float(base + degree_score + token_score + step_score + tie)