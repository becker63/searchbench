from __future__ import annotations

from typing import Mapping

from harness.localization.models import LCAContext, LCATaskIdentity
from harness.loop import IterationRecord, TaskPayload, run_loop
from harness.loop.loop_types import EvaluationMetrics


def _payload(repo: str) -> TaskPayload:
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="owner",
        repo_name="repo",
        base_sha="abc",
    )
    context = LCAContext(issue_title="bug", issue_body="details")
    return TaskPayload(identity=identity, context=context, repo=repo, changed_files=["a.py"])


def test_run_loop_keeps_filesystem_repo_for_ic(monkeypatch, tmp_path):
    ic_repos: list[object] = []

    class DummySpan:
        def __init__(self, name: str = "obs"):
            self.name = name

        def end(self, **kwargs):
            pass

    repo_dir = tmp_path / "ic_repo"
    repo_dir.mkdir()

    monkeypatch.setattr("harness.loop.index_folder", lambda repo: {"repo": "local/fake-repo-id"})
    monkeypatch.setattr("harness.loop.load_policy", lambda: lambda *_: 1.0)

    def fake_run_ic_iteration(task: Mapping[str, object], *args, **kwargs):
        ic_repos.append(task.get("repo"))
        return {"observations": [{"tool": "x"}], "node_count": 1}

    monkeypatch.setattr("harness.loop.run_ic_iteration", fake_run_ic_iteration)
    monkeypatch.setattr("harness.loop.score", lambda result: {"score": 1.0, "ic_nodes": 1, "jc_nodes": 0})
    monkeypatch.setattr("harness.loop.load_tests", lambda repo: [])
    monkeypatch.setattr("harness.loop.format_tests_for_prompt", lambda tests: "")
    monkeypatch.setattr("harness.loop.load_scoring_types", lambda repo: "")
    monkeypatch.setattr("harness.loop.load_scoring_examples", lambda repo: "")
    monkeypatch.setattr("harness.loop.format_scoring_context", lambda *a, **k: "")
    monkeypatch.setattr("harness.loop._read_policy", lambda *a, **k: "def score(node, state):\n    return 0.0\n")
    monkeypatch.setattr("harness.loop._write_policy", lambda *a, **k: None)
    monkeypatch.setattr("harness.loop._hash_tests_dir", lambda *a, **k: "hash")
    monkeypatch.setattr("harness.loop.generate_policy", lambda **kwargs: "def score(node, state):\n    return 1.0\n")

    class DummyPipeline:
        def run(self, repo_root, observation=None, allow_no_parent=False):
            return []

    monkeypatch.setattr("harness.loop.default_pipeline", lambda: DummyPipeline())
    monkeypatch.setattr("harness.loop.find_repo_root", lambda: tmp_path)
    monkeypatch.setattr("harness.loop.start_observation", lambda *a, **k: DummySpan("span"))
    monkeypatch.setattr("harness.loop.emit_score", lambda *a, **k: None)
    monkeypatch.setattr("harness.loop.flush_langfuse", lambda: None)

    payload = _payload(str(repo_dir))
    run_loop(payload, iterations=1)
    assert ic_repos == [str(repo_dir)]
