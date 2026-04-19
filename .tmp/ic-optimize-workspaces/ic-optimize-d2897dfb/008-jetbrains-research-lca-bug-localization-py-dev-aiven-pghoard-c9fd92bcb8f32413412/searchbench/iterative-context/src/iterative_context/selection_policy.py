from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from iterative_context.graph_models import Graph, GraphNode
from iterative_context.types import SelectionCallable


class SelectionPolicy(Protocol):
    """Deterministic selection policy interface."""

    def score(self, node: GraphNode, graph: Graph, step: int) -> float:
        ...

    def choose(self, candidates: Sequence[GraphNode], graph: Graph, step: int) -> GraphNode:
        ...


class CallableSelectionPolicy:
    """Adapter to treat a plain scoring function as a selection policy."""

    def __init__(self, fn: SelectionCallable):
        self._fn = fn

    def score(self, node: GraphNode, graph: Graph, step: int) -> float:
        try:
            value: object = self._fn(node, graph, step)
        except Exception as e:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Policy crashed: {e}") from e

        if not isinstance(value, (int, float)):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("score must be numeric")

        numeric = float(value)

        if math.isnan(numeric) or math.isinf(numeric):
            raise ValueError("invalid score")

        return numeric

    def choose(self, candidates: Sequence[GraphNode], graph: Graph, step: int) -> GraphNode:
        return max(candidates, key=lambda n: (self.score(n, graph, step), n.id))


def wrap_selection_callable(fn: SelectionCallable) -> CallableSelectionPolicy:
    return CallableSelectionPolicy(fn)


__all__ = [
    "SelectionPolicy",
    "CallableSelectionPolicy",
    "wrap_selection_callable",
]
