from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .policy_loader import load_policy
from .runner import run_both
from .scorer import score
from .writer import generate_policy

_POLICY_PATH = Path(__file__).with_name("policy.py")


def _read_policy() -> str:
    return _POLICY_PATH.read_text(encoding="utf-8")


def _write_policy(code: str) -> None:
    _ = _POLICY_PATH.write_text(code, encoding="utf-8")


def run_loop(task: Mapping[str, object], iterations: int = 5) -> list[dict[str, float | int]]:
    """
    Closed-loop optimizer: execute, score, rewrite policy, repeat.
    """
    history: list[dict[str, float | int]] = []

    for _ in range(iterations):
        score_fn = load_policy()
        result = run_both(dict(task), score_fn)
        metrics = score(result)

        current_code = _read_policy()
        new_code = generate_policy(metrics, current_code)
        _write_policy(new_code)

        history.append(metrics)

    return history
