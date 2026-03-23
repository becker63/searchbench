from pathlib import Path

import pytest

from harness.policy_loader import load_policy
from harness.loop import _write_policy, _read_policy


@pytest.fixture(autouse=True)
def restore_policy():
    policy_path = Path(__file__).resolve().parents[2] / "harness" / "policy.py"
    original = _read_policy(policy_path)
    yield
    _write_policy(original, policy_path)


def test_policy_cannot_import_os():
    policy_path = Path(__file__).resolve().parents[2] / "harness" / "policy.py"
    malicious = "import os\n\ndef score(node, state):\n    return 1.0\n"
    _write_policy(malicious, policy_path)
    with pytest.raises(ImportError):
        load_policy()


def test_policy_cannot_use_open():
    policy_path = Path(__file__).resolve().parents[2] / "harness" / "policy.py"
    malicious = "def score(node, state):\n    open('x', 'w')\n    return 1.0\n"
    _write_policy(malicious, policy_path)
    scorer = load_policy()
    with pytest.raises(Exception):
        scorer(None, {})


def test_policy_timeout():
    policy_path = Path(__file__).resolve().parents[1] / "policy.py"
    malicious = "def score(node, state):\n    while True:\n        pass\n"
    _write_policy(malicious, policy_path)
    scorer = load_policy()
    with pytest.raises(TimeoutError):
        scorer(None, {})
