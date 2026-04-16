from __future__ import annotations

import pytest
from pathlib import Path

from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory
from harness.localization.runtime.execute import (
    _resolve_repo_path,
    _run_runner,
    _score_task,
    run_localization_task,
)
from harness.localization.models import LCAContext, LCAGold, LCAPrediction, LCATask, LCATaskIdentity


def _task(repo: Path) -> LCATask:
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    context = LCAContext(issue_title="bug", issue_body="body")
    gold = LCAGold(changed_files=["a.py"])
    return LCATask(identity=identity, context=context, gold=gold, repo=str(repo))


def test_resolve_repo_path_missing(tmp_path):
    task = _task(tmp_path / "missing")
    with pytest.raises(LocalizationEvaluationError) as exc:
        _resolve_repo_path(task, None)
    assert exc.value.category == LocalizationFailureCategory.MATERIALIZATION


def test_run_runner_failure_is_typed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    task = _task(repo)

    def failing_runner(*_a, **_k):
        raise RuntimeError("boom")

    with pytest.raises(LocalizationEvaluationError) as exc:
        _run_runner(task, repo, None, failing_runner)
    assert exc.value.category == LocalizationFailureCategory.RUNNER


def test_score_task_failure_is_typed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    task = _task(repo)

    def bad_eval(*_a, **_k):
        raise ValueError("score fail")

    monkeypatch.setattr("harness.localization.runtime.execute.build_localization_score_eval_record", bad_eval)
    with pytest.raises(LocalizationEvaluationError) as exc:
        _score_task(task, LCAPrediction(predicted_files=["a.py"]), repo)
    assert exc.value.category == LocalizationFailureCategory.SCORING
