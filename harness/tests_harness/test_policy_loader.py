from __future__ import annotations

import pytest

from harness import policy_loader


def test_adapt_score_fn_accepts_supported_arities():
    def score_one(node):
        return 1.0

    def score_two(node, state):
        return 2.0

    def score_three(node, state, ctx):
        return 3.0

    assert policy_loader.adapt_score_fn(score_one)("n", None, None) == 1.0
    assert policy_loader.adapt_score_fn(score_two)("n", "s", None) == 2.0
    assert policy_loader.adapt_score_fn(score_three)("n", "s", "c") == 3.0


def test_adapt_score_fn_rejects_unsupported_arities():
    def score_four(a, b, c, d):
        return 4.0

    with pytest.raises(RuntimeError):
        policy_loader.adapt_score_fn(score_four)
