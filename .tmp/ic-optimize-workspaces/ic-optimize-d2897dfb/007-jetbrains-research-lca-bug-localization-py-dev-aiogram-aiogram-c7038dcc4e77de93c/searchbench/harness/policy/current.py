def frontier_priority(node: GraphNode, graph: Graph, step: int) -> float:
    """Deterministic priority for frontier nodes.

    The function combines a few lightweight, stable signals:

    * **State weight** – prefers *pending* nodes to encourage exploration,
      gives a modest boost to *anchor* nodes and a small baseline for
      *resolved* nodes. *Pruned* nodes receive a massive negative penalty
      to keep them out of the frontier.
    * **Degree signal** – the sum of in‑ and out‑degree capped to avoid
      runaway scores. Nodes with many connections are slightly favoured
      because they are more central.
    * **Token contribution** – for *anchor* and *resolved* nodes, a
      token count (capped) nudges the score upward.
    * **Step encouragement** – a gently increasing term that favours
      later steps, helping the search progress without causing large
      fluctuations.
    """

    # -----------------------------------------------------------------
    # Base weight according to node state
    # -----------------------------------------------------------------
    state = getattr(node, "state", None)
    total: float = 0.0

    if state == "pending":
        total += 8.0
    elif state == "anchor":
        total += 6.0
    elif state == "resolved":
        total += 2.0
    elif state == "pruned":
        # Extremely negative to exclude pruned nodes from consideration
        return -1e9

    # -----------------------------------------------------------------
    # Structural signal – degree (capped for stability)
    # -----------------------------------------------------------------
    try:
        deg = graph.in_degree(node.id) + graph.out_degree(node.id)
    except Exception:
        deg = 0
    total += min(float(deg), 20.0) * 0.4

    # -----------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes
    # -----------------------------------------------------------------
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", 0) or 0
        if isinstance(tokens, (int, float)):
            total += min(float(tokens), 100.0) * 0.08

    # -----------------------------------------------------------------
    # Step‑based encouragement (softly increasing)
    # -----------------------------------------------------------------
    total += min(float(step), 10.0) * 0.04

    return float(total)