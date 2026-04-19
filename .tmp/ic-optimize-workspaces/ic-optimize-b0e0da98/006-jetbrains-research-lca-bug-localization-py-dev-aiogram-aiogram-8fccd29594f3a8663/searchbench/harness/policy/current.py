def frontier_priority(node: GraphNode, graph: Graph, step: int) -> float:
    """Deterministic priority for a graph node.

    The score combines inexpensive, always‑available signals:

    1. Base weight based on the node's ``state``.
    2. Structural weight proportional to the node's total degree
       (in‑degree + out‑degree), capped for stability.
    3. Token weight for ``anchor`` and ``resolved`` nodes, limited to
       a safe maximum.
    4. Step weight that gently favours later traversal steps.
    """

    # -----------------------------------------------------------------
    # 1. Base weight derived from the node's state.
    # -----------------------------------------------------------------
    state = getattr(node, "state", None)
    score: float = 0.0
    if state == "anchor":
        score += 15.0
    elif state == "pending":
        score += 6.0
    elif state == "resolved":
        score += 3.0
    else:  # ``pruned`` or any unexpected state – make it effectively unselectable.
        score -= 1e6

    # -----------------------------------------------------------------
    # 2. Structural signal – total degree (in + out), capped for stability.
    # -----------------------------------------------------------------
    try:
        degree = graph.in_degree(node.id) + graph.out_degree(node.id)  # type: ignore[attr-defined]
    except Exception:
        degree = 0
    if degree > 40:
        degree = 40
    score += degree * 0.3

    # -----------------------------------------------------------------
    # 3. Token‑based signal for anchor and resolved nodes.
    # -----------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            token_val = float(tokens)
            if token_val > 200:
                token_val = 200
            # Slightly higher coefficient to improve token efficiency.
            score += token_val * 0.1

    # -----------------------------------------------------------------
    # 4. Step‑based encouragement – grows slowly with the traversal step.
    # -----------------------------------------------------------------
    step_factor = step if step < 10 else 10
    score += step_factor * 0.1

    return float(score)