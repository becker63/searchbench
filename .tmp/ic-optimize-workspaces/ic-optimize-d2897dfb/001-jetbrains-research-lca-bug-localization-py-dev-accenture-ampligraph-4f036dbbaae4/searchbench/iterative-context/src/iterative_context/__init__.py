"""Iterative Context public API."""

from .exploration import expand, expand_with_policy, resolve, resolve_and_expand  # noqa: F401
from .selection_policy import wrap_selection_callable  # noqa: F401

__all__ = [
    "resolve",
    "expand",
    "resolve_and_expand",
    "expand_with_policy",
    "wrap_selection_callable",
]
