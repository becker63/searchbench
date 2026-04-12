from __future__ import annotations

import types
import pytest

from harness.policy import load as policy_loader


def test_require_frontier_policy_callable_accepts_three_args():
    def priority(node, graph, step):
        return 1.0

    wrapped = policy_loader._require_frontier_policy_callable(priority)
    assert wrapped("n", "g", 1) == 1.0


def test_require_frontier_policy_callable_rejects_wrong_arity():
    def frontier_two(node, graph):
        return 1.0

    with pytest.raises(RuntimeError):
        policy_loader._require_frontier_policy_callable(frontier_two)


def test_load_frontier_policy_uses_selection_callable(monkeypatch, tmp_path):
    policy_file = tmp_path / "policy.py"
    policy_file.write_text(
        "from iterative_context.graph_models import Graph, GraphNode\n"
        "def frontier_priority(node: GraphNode, graph: Graph, step: int) -> float:\n"
        "    return 42.0\n",
        encoding="utf-8",
    )

    def fake_with_name(self, name):
        return policy_file

    monkeypatch.setattr(policy_loader.Path, "with_name", fake_with_name)
    score_fn = policy_loader.load_frontier_policy()
    assert score_fn("n", "g", 0) == 42.0
