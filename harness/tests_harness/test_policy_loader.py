from __future__ import annotations

from pathlib import Path

import pytest

from harness.policy_loader import load_policy


@pytest.fixture()
def policy_path():
    path = Path(__file__).resolve().parents[2] / "harness" / "policy.py"
    original = path.read_text(encoding="utf-8")
    yield path
    path.write_text(original, encoding="utf-8")


def test_missing_score_function_raises(policy_path: Path):
    policy_path.write_text("def not_score():\n    return 1\n", encoding="utf-8")
    with pytest.raises(AttributeError):
        load_policy()


def test_non_callable_score_raises(policy_path: Path):
    policy_path.write_text("score = 1\n", encoding="utf-8")
    with pytest.raises(AttributeError):
        load_policy()


def test_variable_arity_adapts(policy_path: Path):
    policy_path.write_text(
        "def score(node):\n    return 1.5\n",
        encoding="utf-8",
    )
    scorer = load_policy()
    assert scorer(None, None, None) == 1.5
