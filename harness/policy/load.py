from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Callable, cast

from iterative_context.graph_models import Graph, GraphNode  # type: ignore[import-not-found]
from iterative_context.types import SelectionCallable  # type: ignore[import-not-found]


def _require_frontier_policy_callable(priority_fn: Callable[[GraphNode, Graph, int], float]) -> SelectionCallable:
    sig = inspect.signature(priority_fn)
    if len(sig.parameters) != 3:
        raise RuntimeError(
            f"Unsupported frontier policy arity: expected 3 (node, graph, step), got {len(sig.parameters)}"
        )

    def wrapped(node: GraphNode, graph: Graph, step: int) -> float:
        return float(priority_fn(node, graph, step))

    return wrapped


def load_frontier_policy() -> SelectionCallable:
    """
    Load the current policy module and return its traversal-priority callable aligned to SelectionCallable.
    """
    policy_path = Path(__file__).with_name("current.py")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found at {policy_path}")

    spec = importlib.util.spec_from_file_location("policy", policy_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load policy module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    priority_obj = getattr(module, "frontier_priority", None)
    if not callable(priority_obj):
        raise AttributeError("Policy module must define a callable 'frontier_priority'")

    priority_fn_typed: Callable[[GraphNode, Graph, int], float] = cast(
        Callable[[GraphNode, Graph, int], float], priority_obj
    )
    adapted: SelectionCallable = _require_frontier_policy_callable(priority_fn_typed)
    return adapted
