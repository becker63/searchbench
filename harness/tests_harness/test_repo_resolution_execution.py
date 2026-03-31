from __future__ import annotations

from typing import Any, Mapping

from harness.observability import baselines, experiments
from harness import loop, runner
from harness.loop_types import IterationRecord


def test_baseline_resolves_repo(monkeypatch, tmp_path):
    called: dict[str, object | None] = {"repo": None}

    def fake_run_jc_iteration(task: Mapping[str, object], parent_trace=None):
        called["repo"] = task.get("repo")
        return {"observations": []}

    repo_dir = tmp_path / "small_repo"
    repo_dir.mkdir()
    monkeypatch.setenv("TEST_REPO_SMALL", str(repo_dir))
    monkeypatch.setattr(runner, "run_jc_iteration", fake_run_jc_iteration)
    snap = baselines.compute_baseline_for_item(
        baselines.normalize_dataset_item({"id": "item1", "repo": "small", "symbol": "s", "metadata": {}})
    )
    assert called["repo"] == str(repo_dir)
    # logical repo ref preserved
    assert snap.repo == "small"


def test_ic_optimization_resolves_repo(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_run_loop(task: Mapping[str, object], iterations: int, parent_trace=None, baseline_snapshot=None):
        calls.append(str(task.get("repo", "")))
        return [IterationRecord(iteration=0, metrics={"score": 1.0}, pipeline_passed=True)]

    repo_dir = tmp_path / "medium_repo"
    repo_dir.mkdir()
    monkeypatch.setenv("TEST_REPO_MEDIUM", str(repo_dir))
    monkeypatch.setattr(experiments, "run_loop", fake_run_loop)
    baselines_bundle: Any = baselines.make_baseline_bundle(
        [baselines.BaselineSnapshot(repo="medium", symbol="s", jc_result={}, jc_metrics={}, item_id="item2")]
    )
    experiments._run_ic_optimizations(  # type: ignore[attr-defined,arg-type]
        [baselines.normalize_dataset_item({"id": "item3", "repo": "medium", "symbol": "s", "metadata": {}})],
        baselines_bundle,
        dataset_name=None,
        dataset_version=None,
        iterations=1,
        hosted=False,
    )
    assert calls and calls[0] == str(repo_dir)


def test_run_loop_keeps_filesystem_repo_for_ic(monkeypatch, tmp_path):
    ic_repos: list[object] = []

    class DummySpan:
        def __init__(self, name: str = "obs"):
            self.name = name

        def end(self, **kwargs):
            pass

    repo_dir = tmp_path / "ic_repo"
    repo_dir.mkdir()

    monkeypatch.setattr(loop, "index_folder", lambda repo: {"repo": "local/fake-repo-id"})
    monkeypatch.setattr(loop, "load_policy", lambda: lambda *_: 1.0)

    def fake_run_ic_iteration(task: Mapping[str, object], *args, **kwargs):
        ic_repos.append(task.get("repo"))
        return {"observations": [{"tool": "x"}], "node_count": 1}

    monkeypatch.setattr(loop, "run_ic_iteration", fake_run_ic_iteration)
    monkeypatch.setattr(loop, "score", lambda result: {"score": 1.0, "ic_nodes": 1, "jc_nodes": 0})
    monkeypatch.setattr(loop, "load_tests", lambda repo: [])
    monkeypatch.setattr(loop, "format_tests_for_prompt", lambda tests: "")
    monkeypatch.setattr(loop, "load_scoring_types", lambda repo: "")
    monkeypatch.setattr(loop, "load_scoring_examples", lambda repo: "")
    monkeypatch.setattr(loop, "format_scoring_context", lambda *a, **k: "")
    monkeypatch.setattr(loop, "_read_policy", lambda *a, **k: "def score(node, state):\n    return 0.0\n")
    monkeypatch.setattr(loop, "_write_policy", lambda *a, **k: None)
    monkeypatch.setattr(loop, "_hash_tests_dir", lambda *a, **k: "hash")
    monkeypatch.setattr(loop, "generate_policy", lambda **kwargs: "def score(node, state):\n    return 1.0\n")

    class DummyPipeline:
        def run(self, repo_root, observation=None):
            return []

    monkeypatch.setattr(loop, "default_pipeline", lambda: DummyPipeline())
    monkeypatch.setattr(loop, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(loop, "start_trace", lambda *a, **k: DummySpan("trace"))
    monkeypatch.setattr(loop, "start_span", lambda *a, **k: DummySpan("span"))
    monkeypatch.setattr(loop, "record_score", lambda *a, **k: None)
    monkeypatch.setattr(loop, "flush_langfuse", lambda: None)

    loop.run_loop({"symbol": "s", "repo": str(repo_dir)}, iterations=1)
    assert ic_repos == [str(repo_dir)]
