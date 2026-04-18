from __future__ import annotations

from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.orchestration import run_loop


def _task(repo: str) -> LCATask:
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="owner",
        repo_name="repo",
        base_sha="abc",
    )
    context = LCAContext(issue_title="bug", issue_body="details")
    return LCATask(identity=identity, context=context, gold=LCAGold(changed_files=["a.py"]), repo=repo)


def test_run_loop_keeps_filesystem_repo_for_ic(monkeypatch, tmp_path):
    ic_repos: list[object] = []

    class DummySpan:
        def __init__(self, name: str = "obs"):
            self.name = name

        def end(self, **kwargs):
            pass

        def update(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    repo_dir = tmp_path / "ic_repo"
    repo_dir.mkdir()

    monkeypatch.setattr("harness.orchestration.runtime.index_folder", lambda repo: {"repo": "local/fake-repo-id"})
    monkeypatch.setattr("harness.orchestration.runtime.load_frontier_policy", lambda: lambda *_: 1.0)

    def fake_run_ic_iteration(task: LCATask, *args, **kwargs):
        ic_repos.append(task.repo)
        return {"observations": [{"tool": "x"}], "node_count": 1}

    monkeypatch.setattr("harness.agents.localizer.run_ic_iteration", fake_run_ic_iteration)
    monkeypatch.setattr("harness.orchestration.runtime.frontier_priority", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr("harness.orchestration.runtime.load_tests", lambda repo: [])
    monkeypatch.setattr("harness.orchestration.runtime.format_tests_for_prompt", lambda tests: "")
    from harness.utils.type_loader import FrontierContext

    dummy_ctx = FrontierContext(signature="", graph_models="", types="", examples="", notes="")
    monkeypatch.setattr("harness.orchestration.runtime.build_frontier_context", lambda repo: dummy_ctx)
    monkeypatch.setattr("harness.orchestration.runtime.format_frontier_context", lambda *a, **k: "")
    monkeypatch.setattr("harness.orchestration.runtime._read_policy", lambda *a, **k: "def frontier_priority(node, graph, step):\n    return 0.0\n")
    monkeypatch.setattr("harness.orchestration.runtime._write_policy", lambda *a, **k: None)
    monkeypatch.setattr("harness.orchestration.runtime._hash_tests_dir", lambda *a, **k: "hash")
    monkeypatch.setattr("harness.orchestration.runtime.generate_policy", lambda **kwargs: "def frontier_priority(node, graph, step):\n    return 1.0\n")

    class DummyPipeline:
        def run(self, repo_root, observation=None, allow_no_parent=False):
            return []

    monkeypatch.setattr("harness.orchestration.runtime.default_pipeline", lambda: DummyPipeline())
    monkeypatch.setattr("harness.orchestration.runtime.find_repo_root", lambda: tmp_path)
    monkeypatch.setattr("harness.orchestration.runtime.start_observation", lambda *a, **k: DummySpan("span"))
    monkeypatch.setattr("harness.orchestration.runtime.emit_score", lambda *a, **k: None)
    monkeypatch.setattr("harness.orchestration.runtime.flush_langfuse", lambda: None)

    task = _task(str(repo_dir))
    run_loop(task, iterations=1)
    assert ic_repos == [str(repo_dir)]
