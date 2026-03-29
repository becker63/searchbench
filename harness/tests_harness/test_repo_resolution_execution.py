from __future__ import annotations

from typing import Any, Mapping

from harness.observability import baselines, experiments
from harness import runner


def test_baseline_resolves_repo(monkeypatch, tmp_path):
    called = {"repo": None}

    def fake_run_jc_iteration(task: Mapping[str, object], parent_trace=None):
        called["repo"] = task.get("repo")
        return {"observations": []}

    repo_dir = tmp_path / "small_repo"
    repo_dir.mkdir()
    monkeypatch.setenv("TEST_REPO_SMALL", str(repo_dir))
    monkeypatch.setattr(runner, "run_jc_iteration", fake_run_jc_iteration)
    snap = baselines.compute_baseline_for_item({"repo": "small", "symbol": "s"})  # type: ignore[arg-type]
    assert called["repo"] == str(repo_dir)
    # logical repo ref preserved
    assert snap["repo"] == "small"


def test_ic_optimization_resolves_repo(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_run_loop(task: Mapping[str, object], iterations: int, parent_trace=None, baseline_snapshot=None):
        calls.append(task.get("repo", ""))
        return [{"score": 1.0}]

    repo_dir = tmp_path / "medium_repo"
    repo_dir.mkdir()
    monkeypatch.setenv("TEST_REPO_MEDIUM", str(repo_dir))
    monkeypatch.setattr(experiments, "run_loop", fake_run_loop)
    baselines_bundle: Any = {"items": [{"repo": "medium", "symbol": "s", "jc_result": {}, "jc_metrics": {}}]}
    experiments._run_ic_optimizations(  # type: ignore[attr-defined]
        [{"repo": "medium", "symbol": "s"}],
        baselines_bundle,
        dataset_name=None,
        dataset_version=None,
        iterations=1,
        hosted=False,
    )
    assert calls and calls[0] == str(repo_dir)
