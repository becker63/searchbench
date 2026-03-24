def score(node, state):
    """
    Deterministic scoring for a graph node.

    Parameters
    ----------
    node : dict or object
        Graph node containing ``state``, ``id`` and optional ``tokens``.
    state : object
        Typically a graph providing ``in_degree`` and ``out_degree`` methods.
        If absent or lacking these methods, degree is treated as 0.

    Returns
    -------
    float
        Higher values indicate higher priority for expansion.
    """
    # -------------------------------------------------
    # Extract basic fields from ``node`` (dict or attribute style)
    # -------------------------------------------------
    if isinstance(node, dict):
        node_state = node.get("state")
        node_id = node.get("id")
        tokens = node.get("tokens")
    else:
        node_state = getattr(node, "state", None)
        node_id = getattr(node, "id", None)
        tokens = getattr(node, "tokens", None)

    # -------------------------------------------------
    # Base score based on node state
    # -------------------------------------------------
    score_val = 0.0
    if node_state == "pending":
        score_val += 5.0
    if node_state == "anchor":
        score_val += 10.0

    # -------------------------------------------------
    # Add degree information if the ``state`` object provides it
    # -------------------------------------------------
    degree = 0
    try:
        # ``state`` is expected to behave like a graph with in_degree/out_degree
        if hasattr(state, "in_degree") and hasattr(state, "out_degree"):
            if node_id is not None:
                degree = state.in_degree(node_id) + state.out_degree(node_id)
    except Exception:
        # Any failure (e.g., wrong type) falls back to degree = 0
        degree = 0
    score_val += degree * 0.5

    # -------------------------------------------------
    # Token contribution, capped at 50 tokens.
    # -------------------------------------------------
    token_count = 0
    if isinstance(tokens, (int, float)):
        token_count = tokens
    elif isinstance(tokens, (list, tuple, set)):
        token_count = len(tokens)

    if token_count:
        # Ensure token count is non‑negative and respect the 50‑token cap
        token_count = max(0, token_count)
        score_val += min(token_count, 50) * 0.1

    return float(score_val)