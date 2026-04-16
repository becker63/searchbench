from __future__ import annotations

import time

import pytest

from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory
from harness.localization.runtime.evaluate import LocalizationEvaluationResult, LocalizationEvaluationTaskResult
from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.localization.scoring_models import ScoreContext, ScoreEngine, summarize_batch_scores
from harness.telemetry.hosted import experiments


def _task(repo: str) -> LCATask:
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
    return LCATask(identity=identity, context=context, gold=gold, repo=repo)


def _eval_result(task: LCATask, prediction: list[str] | None = None) -> LocalizationEvaluationResult:
    prediction = prediction or ["a.py"]
    bundle = ScoreEngine().evaluate(
        ScoreContext(
            predicted_files=tuple(prediction),
            gold_files=("a.py",),
            gold_min_hops=0,
            issue_min_hops=0,
        )
    )
    summary = summarize_batch_scores({task.task_id: bundle})
    return LocalizationEvaluationResult(
        score_summary=summary,
        machine_score=summary.aggregate_composed_score,
        items=[
            LocalizationEvaluationTaskResult(
                task=task,
                score_bundle=bundle,
                prediction=prediction,
                evidence=None,
                materialization=None,
                trace_id=None,
            )
        ],
        failure=None,
    )


def test_experiment_uses_shared_backend(monkeypatch, tmp_path) -> None:
    task = _task(str(tmp_path / "repo"))
    calls = []

    def fake_backend(**kwargs):
        calls.append(kwargs)
        return _eval_result(task)

    monkeypatch.setattr(experiments, "evaluate_localization_batch", fake_backend)
    monkeypatch.setattr(
        experiments,
        "_resolve_items",
        lambda req: [task],
    )
    req = experiments.HostedLocalizationRunRequest(
        dataset="d",
        version="v1",
        dataset_config="c",
        dataset_split="s",
        session=None,
    )
    result = experiments.run_hosted_localization_experiment(req, materializer=None)
    assert calls
    assert result.items[0].prediction == ["a.py"]


def test_experiment_parallel_preserves_order(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    t1 = _task(str(repo_root / "r1"))
    t2 = _task(str(repo_root / "r2"))
    tasks = [t2, t1]  # intentionally unsorted to exercise deterministic selection
    durations = {t1.task_id: 0.05, t2.task_id: 0.01}

    def fake_backend(**kwargs):
        task = kwargs["tasks"][0]
        time.sleep(durations[task.task_id])
        return _eval_result(task, [task.task_id])

    monkeypatch.setattr(experiments, "evaluate_localization_batch", fake_backend)
    req = experiments.HostedLocalizationRunRequest(
        dataset="d",
        version="v1",
        dataset_config="c",
        dataset_split="s",
        session=None,
        max_workers=2,
    )
    result = experiments.run_hosted_localization_experiment(req, materializer=None, tasks=tasks)
    assert [item.prediction[0] for item in result.items] == [t1.task_id, t2.task_id]
    assert result.failure is None


def test_experiment_failure_is_deterministic(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    t1 = _task(str(repo_root / "r1"))
    t2 = _task(str(repo_root / "r2"))
    calls = []

    def fake_backend(**kwargs):
        task = kwargs["tasks"][0]
        calls.append(task.task_id)
        if task.task_id == t2.task_id:
            raise LocalizationEvaluationError(LocalizationFailureCategory.RUNNER, "runner failed", task_id=task.task_id)
        return _eval_result(task, [task.task_id])

    monkeypatch.setattr(experiments, "evaluate_localization_batch", fake_backend)
    req = experiments.HostedLocalizationRunRequest(
        dataset="d",
        version="v1",
        dataset_config="c",
        dataset_split="s",
        session=None,
        max_workers=2,
    )
    result = experiments.run_hosted_localization_experiment(req, materializer=None, tasks=[t1, t2])
    assert result.failure is not None
    assert result.failure.task_id == t2.task_id
    # Only the successful task (t1) should be in results, ordered
    assert [item.prediction[0] for item in result.items] == [t1.task_id]


def test_experiment_serial_worker_count_one(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    t1 = _task(str(repo_root / "r1"))
    t2 = _task(str(repo_root / "r2"))
    calls = []

    def fake_backend(**kwargs):
        task = kwargs["tasks"][0]
        calls.append(task.task_id)
        return _eval_result(task, [task.task_id])

    monkeypatch.setattr(experiments, "evaluate_localization_batch", fake_backend)
    req = experiments.HostedLocalizationRunRequest(
        dataset="d",
        version="v1",
        dataset_config="c",
        dataset_split="s",
        session=None,
        max_workers=1,
    )
    result = experiments.run_hosted_localization_experiment(req, materializer=None, tasks=[t2, t1])
    assert calls == [t1.task_id, t2.task_id]
    assert [item.prediction[0] for item in result.items] == [t1.task_id, t2.task_id]
    assert result.failure is None


def test_missing_parent_trace_guard(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    task = _task(str(repo_root / "r1"))
    with pytest.raises(experiments._TaskFailure):
        experiments._run_experiment_task(task=task, parent_trace=None, materializer=None, session_id=None)
