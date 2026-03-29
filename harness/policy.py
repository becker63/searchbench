def score(node: object, state: object, context: object | None = None) -> float:
    """Deterministic scoring function.

    Parameters
    ----------
    node: object
        Graph node, expected to have ``state``, ``id`` and optional ``tokens`` attributes.
    state: object
        Graph‑like object providing ``in_degree`` and ``out_degree`` callables.
    context: object | None, optional
        Additional context; if it provides a numeric ``scale`` attribute it will be applied.
    """
    # Safely extract attributes.
    n_state = getattr(node, "state", None)
    n_id = getattr(node, "id", None)
    tokens = getattr(node, "tokens", None)

    total = 0.0

    # State contribution.
    if n_state == "pending":
        total += 5.0
    if n_state == "anchor":
        total += 10.0

    # Degree contribution.
    if n_id is not None:
        degree = 0
        in_deg_func = getattr(state, "in_degree", None)
        out_deg_func = getattr(state, "out_degree", None)
        if callable(in_deg_func):
            try:
                val = in_deg_func(n_id)
                if isinstance(val, (int, float)):
                    degree += int(val)
            except Exception:
                pass
        if callable(out_deg_func):
            try:
                val = out_deg_func(n_id)
                if isinstance(val, (int, float)):
                    degree += int(val)
            except Exception:
                pass
        total += degree * 0.5

    # Tokens contribution.
    if isinstance(tokens, (int, float)):
        total += min(tokens, 50) * 0.1

    # Optional scaling from context.
    if context is not None:
        scale = getattr(context, "scale", None)
        if isinstance(scale, (int, float)):
            total *= float(scale)

    return float(total)