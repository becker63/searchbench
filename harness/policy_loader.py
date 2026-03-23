from __future__ import annotations

import builtins
import math
import signal
from pathlib import Path
from typing import Callable, cast

_BANNED_IMPORTS = {"os", "sys", "subprocess", "socket", "shutil", "pathlib"}
_ALLOWED_IMPORTS = {"math"}
_ALLOWED_BUILTINS: dict[str, object] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "float": float,
    "int": int,
    "bool": bool,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "RuntimeError": RuntimeError,
    "__build_class__": builtins.__build_class__,
    "__name__": "policy",
}


def _safe_import(
    name: str,
    globals: dict[str, object] | None = None,
    locals: dict[str, object] | None = None,
    fromlist: tuple[str, ...] | list[str] = (),
    level: int = 0,
) -> object:
    root = name.split(".")[0]
    if root in _BANNED_IMPORTS or (root not in _ALLOWED_IMPORTS and root != ""):
        raise ImportError(f"Import of '{name}' is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _build_safe_globals() -> dict[str, object]:
    safe_builtins: dict[str, object] = dict(_ALLOWED_BUILTINS)
    safe_builtins["__import__"] = _safe_import  # type: ignore[assignment]
    safe_globals: dict[str, object] = {"__builtins__": safe_builtins, "math": math}
    return safe_globals


def _enforce_timeout(func: Callable[[object, object], float], seconds: int = 2) -> Callable[[object, object], float]:
    def wrapped(node: object, state: object) -> float:
        def handler(signum, frame):  # noqa: ANN001, ARG001
            raise TimeoutError("Policy execution timed out")

        previous = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            return float(func(node, state))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    return wrapped


def load_policy() -> Callable[[object, object], float]:
    """
    Load the current policy module with restricted execution and return its score callable.
    """
    policy_path = Path(__file__).with_name("policy.py")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found at {policy_path}")

    safe_globals = _build_safe_globals()
    code = policy_path.read_text(encoding="utf-8")

    compiled = compile(code, str(policy_path), "exec")
    exec(compiled, safe_globals, safe_globals)  # noqa: S102

    score_fn_obj = safe_globals.get("score")
    if not callable(score_fn_obj):
        raise AttributeError("Policy module must define a callable 'score(node, state)'")

    score_fn_typed: Callable[[object, object], float] = cast(Callable[[object, object], float], score_fn_obj)
    return _enforce_timeout(score_fn_typed)
