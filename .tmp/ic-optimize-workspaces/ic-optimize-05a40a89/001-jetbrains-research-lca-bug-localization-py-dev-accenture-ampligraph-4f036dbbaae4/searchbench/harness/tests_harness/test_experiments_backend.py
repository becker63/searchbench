from __future__ import annotations

import time

import pytest

from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory
from harness.localization.runtime.evaluate import (
    LocalizationEvaluationResult,
    LocalizationEvaluationTaskResult,
    prediction_filenames,
)
from harness.localization.models import (
    LCAContext,
    LCAGold,
    LCATask,
    LCATaskIdentity,
    LocalizationPrediction,
)
from harness.scoring import ScoreContext, ScoreEngine, summarize_batch_scores
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


def _eval_result(
    task: LCATask,
    prediction: LocalizationPrediction | None = None,
) -> LocalizationEvaluationResult:
    prediction = prediction or LocalizationPrediction(predicted_files=["a.py"])
    bundle = ScoreEngine().evaluate(
        ScoreContext(
            predicted_files=tuple(prediction.predicted_files),
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
        session=None,
    )
    result = experiments.run_hosted_localization_experiment(req, worktree_manager=None)
    assert calls
    assert isinstance(calls[0]["tasks"][0], LCATask)
    assert calls[0]["dataset_provenance"] == "langfuse"
    assert result.items[0].prediction == ["a.py"]


def test_experiment_dataset_aggregates_emit_component_scores(monkeypatch, tmp_path):
    task = _task(str(tmp_path / "repo"))
    eval_result = _eval_result(task)
    item = experiments.LocalizationRunItemResult(
        task=task,
        score_summary=eval_result.items[0].score_bundle.model_copy(
            update={"item_id": task.task_id}
        ),
        prediction=prediction_filenames(eval_result.items[0].prediction),
    )
    emitted: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []

    class DummySpan:
        id = "dataset-span"

        def score_trace(self, **kwargs):
            emitted.append(kwargs)

        def update(self, **kwargs):
            updates.append(kwargs)

    experiments._emit_experiment_dataset_aggregates(DummySpan(), [item])

    names = {item["name"] for item in emitted}
    assert {
        "experiment.aggregate_composed_score",
        "experiment.aggregate_gold_hop",
        "experiment.aggregate_issue_hop",
    } <= names
    assert updates
    assert updates[-1]["metadata"]["score_summary"]["aggregate_composed_score"] == 1.0


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
        return _eval_result(task, LocalizationPrediction(predicted_files=[task.task_id]))

    monkeypatch.setattr(experiments, "evaluate_localization_batch", fake_backend)
    req = experiments.HostedLocalizationRunRequest(
        dataset="d",
        version="v1",
        session=None,
        max_workers=2,
    )
    result = experiments.run_hosted_localization_experiment(req, worktree_manager=None, tasks=tasks)
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
        return _eval_result(task, LocalizationPrediction(predicted_files=[task.task_id]))

    monkeypatch.setattr(experiments, "evaluate_localization_batch", fake_backend)
    req = experiments.HostedLocalizationRunRequest(
        dataset="d",
        version="v1",
        session=None,
        max_workers=2,
    )
    result = experiments.run_hosted_localization_experiment(req, worktree_manager=None, tasks=[t1, t2])
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
        return _eval_result(task, LocalizationPrediction(predicted_files=[task.task_id]))

    monkeypatch.setattr(experiments, "evaluate_localization_batch", fake_backend)
    req = experiments.HostedLocalizationRunRequest(
        dataset="d",
        version="v1",
        session=None,
        max_workers=1,
    )
    result = experiments.run_hosted_localization_experiment(req, worktree_manager=None, tasks=[t2, t1])
    assert calls == [t1.task_id, t2.task_id]
    assert [item.prediction[0] for item in result.items] == [t1.task_id, t2.task_id]
    assert result.failure is None


def test_missing_parent_trace_guard(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    task = _task(str(repo_root / "r1"))
    with pytest.raises(experiments._TaskFailure):
        experiments._run_experiment_task(task=task, parent_trace=None, worktree_manager=None, session_id=None)
