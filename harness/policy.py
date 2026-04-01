def _get_attr(obj: object, name: str, default: object = None) -> object:
    """Safely retrieve an attribute.
    Returns ``default`` if the attribute lookup fails.
    """
    try:
        return getattr(obj, name)
    except Exception:
        return default


def score(node: object, state: object, context: object | None = None) -> float:
    """Deterministic scoring function.

    Parameters
    ----------
    node: object
        Expected to have ``state``, ``id`` and optional ``tokens`` attributes.
    state: object
        Expected to provide ``in_degree`` and ``out_degree`` callables.
    context: object | None, optional
        If provided and contains a numeric ``scale`` attribute, the total is multiplied by it.
    """
    n_state = _get_attr(node, "state")
    n_id = _get_attr(node, "id")
    tokens = _get_attr(node, "tokens")

    total: float = 0.0

    # State contribution.
    if n_state == "pending":
        total += 5.0
    if n_state == "anchor":
        total += 10.0

    # Degree contribution.
    if n_id is not None:
        degree = 0
        in_func = _get_attr(state, "in_degree")
        out_func = _get_attr(state, "out_degree")
        if callable(in_func):
            try:
                val = in_func(n_id)
                if isinstance(val, (int, float)):
                    degree += int(val)
            except Exception:
                pass
        if callable(out_func):
            try:
                val = out_func(n_id)
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
        scale = _get_attr(context, "scale")
        if isinstance(scale, (int, float)):
            total *= float(scale)

    return float(total)