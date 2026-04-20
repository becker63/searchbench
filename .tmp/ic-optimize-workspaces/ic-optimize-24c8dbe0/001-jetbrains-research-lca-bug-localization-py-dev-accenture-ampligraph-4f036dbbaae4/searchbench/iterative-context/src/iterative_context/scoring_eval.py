from __future__ import annotations

from collections.abc import Callable

from iterative_context.graph_models import Graph
from iterative_context.scoring import default_score_fn
from iterative_context.traversal import DefaultExpansionPolicy, run_traversal
from iterative_context.types import SelectionCallable


def run_with_scoring(
    graph_factory: Callable[[], Graph],
    score_fn: SelectionCallable | None,
    steps: int,
) -> Graph:
    """Execute traversal with an injected scoring callable."""
    graph = graph_factory()
    return run_traversal(
        graph,
        steps=steps,
        expansion_policy=DefaultExpansionPolicy(),
        score_fn=score_fn or default_score_fn,
    )


__all__ = ["run_with_scoring"]
