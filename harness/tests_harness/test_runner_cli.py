from __future__ import annotations

import importlib
from importlib import util as importlib_util

from harness.loop_types import IterationRecord
from harness.observability.baselines import BaselineBundle

def test_root_shim_invokes_harness_main(monkeypatch):
    harness_run = importlib.import_module("harness.run")
    called = {"main": False}

    def fake_main(argv=None):
        called["main"] = True

    monkeypatch.setattr(harness_run, "main", fake_main)
    from pathlib import Path

    root_path = Path(__file__).resolve().parents[2] / "run.py"
    spec = importlib_util.spec_from_file_location("root_run", root_path)
    assert spec and spec.loader
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    module.main()
    assert called["main"]


def test_ad_hoc_runner_resolves_repo(monkeypatch, tmp_path, capsys):
    harness_run = importlib.import_module("harness.run")
    repo_dir = tmp_path / "small_repo"
    repo_dir.mkdir()
    monkeypatch.setenv("TEST_REPO_SMALL", str(repo_dir))
    monkeypatch.setattr(
        "harness.run.run_loop",
        lambda task, iterations=3: [IterationRecord(iteration=0, metrics={"score": 1.0})],
        raising=False,
    )
    harness_run.main([])
    out = capsys.readouterr().out
    assert "small_repo" in out
    assert "expand_node" in out or "symbol" in out


def test_ad_hoc_runner_prints_failed_step(monkeypatch, tmp_path, capsys):
    harness_run = importlib.import_module("harness.run")
    repo_dir = tmp_path / "small_repo"
    repo_dir.mkdir()
    monkeypatch.setenv("TEST_REPO_SMALL", str(repo_dir))

    history = [
        IterationRecord(
            iteration=0,
            metrics={"score": 0.0},
            pipeline_passed=False,
            failed_step="basedpyright",
            failed_exit_code=1,
            failed_summary="type errors here",
        )
    ]
    monkeypatch.setattr("harness.run.run_loop", lambda task, iterations=3: history, raising=False)
    harness_run.main([])
    out = capsys.readouterr().out
    assert "Last failed step: basedpyright" in out
    assert "Exit code: 1" in out
    assert "Failure summary: type errors here" in out


def test_hosted_baseline_flow_calls_experiment(monkeypatch, tmp_path, capsys):
    harness_run = importlib.import_module("harness.run")
    called = {"jc": False}

    def fake_run_hosted_jc_baseline_experiment(name, version=None):
        called["jc"] = True
        return BaselineBundle(items=[])

    monkeypatch.setattr(harness_run, "run_hosted_jc_baseline_experiment", fake_run_hosted_jc_baseline_experiment)
    argv = ["--dataset", "ds", "--jc-baseline"]
    harness_run.main(argv)
    assert called["jc"]
