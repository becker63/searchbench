from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Callable, cast

from iterative_context.graph_models import Graph, GraphNode  # type: ignore[import-not-found]
from iterative_context.types import SelectionCallable  # type: ignore[import-not-found]


def _require_selection_callable(score_fn: Callable[[GraphNode, Graph, int], float]) -> SelectionCallable:
    sig = inspect.signature(score_fn)
    if len(sig.parameters) != 3:
        raise RuntimeError(f"Unsupported score function arity: expected 3 (node, graph, step), got {len(sig.parameters)}")

    def wrapped(node: GraphNode, graph: Graph, step: int) -> float:
        return float(score_fn(node, graph, step))

    return wrapped


def load_policy() -> SelectionCallable:
    """
    Load the current policy module and return its score callable aligned to SelectionCallable.
    """
    policy_path = Path(__file__).with_name("current.py")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found at {policy_path}")

    spec = importlib.util.spec_from_file_location("policy", policy_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load policy module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    score_fn_obj = getattr(module, "score", None)
    if not callable(score_fn_obj):
        raise AttributeError("Policy module must define a callable 'score'")

    score_fn_typed: Callable[[GraphNode, Graph, int], float] = cast(Callable[[GraphNode, Graph, int], float], score_fn_obj)
    adapted: SelectionCallable = _require_selection_callable(score_fn_typed)
    return adapted
