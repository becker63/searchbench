from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Callable, cast


def load_policy() -> Callable[[object, object], float]:
    """
    Load the current policy module fresh and return its score callable.

    A unique module name is used per load to avoid reusing cached modules.
    """
    policy_path = Path(__file__).with_name("policy.py")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found at {policy_path}")

    module_name = f"harness_policy_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, policy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load policy module from {policy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    score_fn = getattr(module, "score", None)
    if not callable(score_fn):
        raise AttributeError("Policy module must define a callable 'score(node, state)'")

    return cast(Callable[[object, object], float], score_fn)
