from __future__ import annotations

import sys

from harness.localization.evaluation_backend import LocalizationEvaluationResult, LocalizationEvaluationTaskResult
from harness.localization.models import LocalizationMetrics, LCATask
import run as run_module


def test_cli_single_routes_through_backend(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    calls = []

    def fake_backend(req):
        calls.append(req)
        task = req.tasks[0] if req.tasks else LCATask.model_validate({})
        metrics = LocalizationMetrics(precision=1.0, recall=1.0, f1=1.0, hit=1.0, score=1.0)
        return LocalizationEvaluationResult(
            aggregate_score=1.0,
            aggregate_metrics=metrics,
            machine_score=1.0,
            items=[
                LocalizationEvaluationTaskResult(
                    task=task,
                    metrics=metrics,
                    prediction=["a.py"],
                    evidence=None,
                    materialization=None,
                    trace_id=None,
                )
            ],
            failure=None,
        )

    monkeypatch.setattr(run_module, "evaluate_localization_batch", fake_backend)
    argv = [
        "--mode",
        "localization-single",
        "--repo-path",
        str(repo_dir),
        "--changed-file",
        "a.py",
    ]
    # Capture stdout to avoid polluting test output
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    run_module.main(argv)
    assert calls, "CLI single-task path should route through shared backend"
