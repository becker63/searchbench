def frontier_priority(node: "GraphNode", graph: "Graph", step: int) -> float:
    """Deterministic priority for a node in the frontier.

    The function combines a base weight derived from the node's ``state`` with
    three deterministic signals:

    * **Structural signal** – based on the total degree (in‑plus‑out) of the node.
    * **Token signal** – for ``anchor`` and ``resolved`` nodes, scaling with the
      number of tokens attached to the node.
    * **Step signal** – a modest encouragement that grows with the traversal
      step index.

    The constants have been tuned to favour ``anchor`` and ``resolved`` nodes
    (which typically improve the gold‑hop and token‑efficiency metrics) while
    keeping the calculation fully deterministic and type‑checked.
    """

    # ------------------------------------------------------------------
    # Base weight according to node state.
    # ------------------------------------------------------------------
    state = node.state
    if state == "anchor":
        base = 15.0
    elif state == "pending":
        base = 8.0
    elif state == "resolved":
        # Resolved nodes are valuable for token efficiency; give them a higher base.
        base = 12.0
    else:  # pruned or any unexpected state.
        # Keep a large negative penalty to ensure such nodes are never selected.
        base = -1_000_000.0

    # ------------------------------------------------------------------
    # Structural signal – degree (in + out).
    # Cap the degree to avoid runaway values and apply a slightly larger
    # multiplier than the original policy.
    # ------------------------------------------------------------------
    deg = graph.in_degree(node.id) + graph.out_degree(node.id)
    deg_capped = deg if deg < 60 else 60
    deg_weight = deg_capped * 0.6

    # ------------------------------------------------------------------
    # Token‑based signal for anchor and resolved nodes.
    # ------------------------------------------------------------------
    token_weight = 0.0
    if state in ("anchor", "resolved"):
        # ``tokens`` may be ``None`` for anchor nodes; treat missing as 0.
        tokens = getattr(node, "tokens", 0) or 0
        token_capped = tokens if tokens < 300 else 300
        token_weight = token_capped * 0.1

    # ------------------------------------------------------------------
    # Step‑based encouragement – grows with step but is capped for stability.
    # ------------------------------------------------------------------
    step_capped = step if step < 20 else 20
    step_weight = step_capped * 0.3

    # ------------------------------------------------------------------
    # Final priority – sum of all components.
    # ------------------------------------------------------------------
    return base + deg_weight + token_weight + step_weight