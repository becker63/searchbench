from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import importlib

import harness.run as cli
from harness.entrypoints import HostedICOptimizationRequest
from harness.observability.datasets import DatasetItem
from harness.observability.session_policy import SessionConfig
from harness.observability.baselines import BaselineBundle, BaselineSnapshot


def test_cli_ad_hoc_dispatch(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "flush_langfuse", lambda: None, raising=False)
    monkeypatch.setattr(cli, "resolve_repo_target", lambda repo: f"resolved:{repo}", raising=False)

    def fake_run_loop(task, iterations, parent_trace=None, baseline_snapshot=None, session_id=None):
        captured.update(
            task=task,
            iterations=iterations,
            session_id=session_id,
        )
        return []

    monkeypatch.setattr(cli, "run_loop", fake_run_loop, raising=False)
    cli.main(["--repo", "foo", "--symbol", "bar", "--iterations", "2", "--session-id", "sess-123"])

    assert captured["task"]["repo"] == "resolved:foo"
    assert captured["task"]["symbol"] == "bar"
    assert captured["iterations"] == 2
    assert captured["session_id"] == "sess-123"


def test_hosted_opt_calls_run_loop_with_item_session(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_run_loop(task, iterations, parent_trace=None, baseline_snapshot=None, session_id=None):
        calls.append(
            {"task": task, "iterations": iterations, "session_id": session_id, "baseline": baseline_snapshot}
        )
        return []

    # prepare dataset items and baseline
    item = DatasetItem(id="item1", repo="r", symbol="s")
    baseline = BaselineSnapshot(repo="r", symbol="s", jc_result={}, jc_metrics={}, item_id="item1")
    bundle = BaselineBundle(items=[baseline])

    import harness.observability.experiments as experiments

    monkeypatch.setattr(experiments, "fetch_dataset_items", lambda name, version=None: [item])
    monkeypatch.setattr(experiments, "run_loop", fake_run_loop)
    monkeypatch.setattr(experiments, "resolve_repo_target", lambda repo: f"resolved:{repo}")
    import harness.observability.score_emitter as score_emitter

    monkeypatch.setattr(score_emitter, "emit_score_for_handle", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiments, "emit_score_for_handle", lambda *args, **kwargs: None)
    from harness.observability.runner_integration import HostedRunItemResult, HostedRunResult

    def fake_hosted_dataset_experiment(*args, **kwargs):
        task_fn = kwargs["task_fn"]
        output = task_fn(item.model_dump(), None)
        return HostedRunResult(items=[HostedRunItemResult(item=item.model_dump(), output=output)])

    monkeypatch.setattr(experiments, "run_hosted_dataset_experiment", fake_hosted_dataset_experiment)

    req = HostedICOptimizationRequest(
        dataset="ds",
        version="v1",
        iterations=1,
        baselines=bundle,
        session=SessionConfig(session_scope="item", allow_generated_fallback=True),
    )
    experiments.run_hosted_ic_optimization_experiment(req)

    assert calls, "run_loop was not invoked"
    assert calls[0]["session_id"] is not None
    assert calls[0]["session_id"]


def test_cli_does_not_import_leaf_ops():
    text = open("run.py", "r", encoding="utf-8").read()
    assert "import runner" not in text
    assert "from harness.runner" not in text
    assert "from .runner" not in text
