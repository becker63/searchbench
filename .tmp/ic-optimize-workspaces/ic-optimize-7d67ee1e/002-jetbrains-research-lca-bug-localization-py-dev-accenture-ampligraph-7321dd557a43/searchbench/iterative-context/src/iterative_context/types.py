from __future__ import annotations

from collections.abc import Callable

from iterative_context.graph_models import Graph, GraphNode

SelectionCallable = Callable[[GraphNode, Graph, int], float]


__all__ = ["SelectionCallable"]
