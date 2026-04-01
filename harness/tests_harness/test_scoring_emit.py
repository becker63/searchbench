from __future__ import annotations

from pathlib import Path

import pytest

import harness.loop as loop_module
from harness.pipeline.types import StepResult
from harness.observability import langfuse as langfuse_module


class DummyPipeline:
    def __init__(self, results: list[StepResult]):
        self._results = results

    def run(self, repo_root: Path, observation=None) -> list[StepResult]:
        return self._results


def test_run_policy_pipeline_emits_step_scores(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_emit_score(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(loop_module, "emit_score", fake_emit_score)
    results = [
        StepResult(name="ruff", success=False, exit_code=1, stdout="", stderr="lint"),
        StepResult(name="pytest_iterative_context", success=True, exit_code=0, stdout="", stderr=""),
    ]
    pipeline = DummyPipeline(results)

    class Obs:
        id = "obs1"
        trace_id = "trace1"
        dataset_run_id = "dsrun1"

    emitted, raw, success = loop_module.run_policy_pipeline(pipeline, Path("."), repair_obs=Obs())
    assert emitted == raw == results
    assert success is False
    names = {c["name"] for c in calls}
    assert "pipeline.ruff.exit_code" in names
    assert "pipeline.ruff.passed" in names
    assert "pipeline.pytest_iterative_context.exit_code" in names
    assert "pipeline.pytest_iterative_context.passed" in names
    for call in calls:
        assert call["trace_id"] == "trace1"
        assert call["dataset_run_id"] == "dsrun1"


def test_emit_score_surfaces_missing_client(monkeypatch):
    monkeypatch.setattr(langfuse_module, "get_langfuse_client", lambda: None)
    with pytest.raises(RuntimeError):
        langfuse_module.emit_score(name="metric", value=1.0, trace_id="t1")
