from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Callable, cast

from pydantic import BaseModel, ConfigDict, ValidationError


class ScoreFunctionConfig(BaseModel):
    """Typed adapter for policy score functions."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    score_fn: Callable[..., float]


def adapt_score_fn(score_fn: Callable[..., float]) -> Callable[[object, object, object], float]:
    try:
        config = ScoreFunctionConfig(score_fn=score_fn)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid score function: {exc}") from exc

    sig = inspect.signature(config.score_fn)
    arity = len(sig.parameters)
    if arity not in (1, 2, 3):
        raise RuntimeError(f"Unsupported score function arity: expected 1-3, got {arity}")

    def wrapped(*call_args: object, **call_kwargs: object) -> float:
        node, state, context = (list(call_args) + [None, None, None])[:3]
        if arity == 3:
            return float(config.score_fn(node, state, context))
        if arity == 2:
            return float(config.score_fn(node, state))
        return float(config.score_fn(node))

    return wrapped


def load_policy() -> Callable[[object, object, object], float]:
    """
    Load the current policy module and return its score callable (adapted to node, state, context).
    """
    policy_path = Path(__file__).with_name("policy.py")
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

    score_fn_typed: Callable[..., float] = cast(Callable[..., float], score_fn_obj)
    adapted: Callable[[object, object, object], float] = adapt_score_fn(score_fn_typed)
    return adapted
