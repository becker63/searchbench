def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority used for frontier node selection.

    The function combines several inexpensive signals that are available
    locally on the ``node`` and the ``graph``.  All arithmetic is performed
    with ``float`` values so the return type is stable across Python
    versions.

    Signals
    -------
    * **Base weight** – varies by ``state``.  ``pending`` nodes are favoured
      because they are the primary expansion targets.  ``anchor`` and
      ``resolved`` nodes receive a smaller boost, while ``pruned`` nodes are
      heavily penalised to keep them out of the frontier.
    * **Degree signal** – the (undirected) degree of the node in the graph
      capped at 20 and scaled by ``0.3``.  Nodes with more connections are
      more likely to open new exploration paths.
    * **Token signal** – for ``anchor`` and ``resolved`` nodes that carry a
      ``tokens`` attribute.  The token count is capped at 100 and scaled by
      ``0.05``.
    * **Step encouragement** – a modest linear increase up to the first
      three steps to break ties deterministically.
    """

    # Base weight according to the node's state.
    state = getattr(node, "state", None)
    if state == "pending":
        base = 12.0
    elif state == "anchor":
        base = 10.0
    elif state == "resolved":
        base = 5.0
    elif state == "pruned":
        # A very large negative value guarantees exclusion from the frontier.
        base = -1e6
    else:
        base = 0.0

    # Structural signal – degree (capped for stability).
    try:
        deg = graph.degree(node.id)  # type: ignore[attr-defined]
    except Exception:
        deg = 0
    deg_signal = min(float(deg), 20.0) * 0.3

    # Token‑based signal for anchor and resolved nodes.
    token_signal = 0.0
    if state in ("anchor", "resolved"):
        tokens = getattr(node, "tokens", None)
        if isinstance(tokens, (int, float)):
            token_signal = min(float(tokens), 100.0) * 0.05

    # Step‑based encouragement – modest linear boost up to step 3.
    step_signal = min(float(step), 3.0) * 0.1

    return float(base + deg_signal + token_signal + step_signal)