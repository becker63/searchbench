from __future__ import annotations

import pytest

from harness.localization.models import (
    LCAContext,
    LCAGold,
    LCATask,
    LCATaskIdentity,
    LocalizationPrediction,
)
from harness.localization.runtime.evaluate import LocalizationEvaluationResult, LocalizationEvaluationTaskResult
from harness.scoring import ScoreBundle, ScoreContext, ScoreEngine, summarize_batch_scores
from harness.orchestration.runtime import evaluate_policy_on_item


def _identity() -> LCATaskIdentity:
    return LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )


def _bundle(gold_hops: int, issue_hops: int) -> ScoreBundle:
    return ScoreEngine().evaluate(
        ScoreContext(
            predicted_files=("a.py",),
            gold_files=("a.py",),
            gold_min_hops=gold_hops,
            issue_min_hops=issue_hops,
        )
    )


def _task(tmp_path) -> LCATask:
    return LCATask(
        identity=_identity(),
        context=LCAContext(issue_title="bug", issue_body="body"),
        gold=LCAGold(changed_files=["a.py"]),
        repo=str(tmp_path),
    )


def test_machine_uses_batch_composed_score(monkeypatch, tmp_path):
    task = _task(tmp_path)
    score_bundle = _bundle(gold_hops=1, issue_hops=1)
    task_result = LocalizationEvaluationTaskResult(
        task=LCATask(
            identity=task.identity,
            context=task.context,
            gold=LCAGold(changed_files=["a.py"]),
            repo=task.repo,
        ),
        score_bundle=score_bundle,
        prediction=LocalizationPrediction(predicted_files=["a.py"]),
    )
    score_summary = summarize_batch_scores({"task": score_bundle})
    eval_result = LocalizationEvaluationResult(
        score_summary=score_summary,
        machine_score=score_summary.aggregate_composed_score,
        items=[task_result],
        failure=None,
    )

    monkeypatch.setattr("harness.orchestration.evaluation.evaluate_localization_batch", lambda **kwargs: eval_result)
    result = evaluate_policy_on_item(task, None, None, 0)

    assert result.control_score == pytest.approx(score_summary.aggregate_composed_score)
    assert result.metrics.get("score") == pytest.approx(score_bundle.composed_score)


def test_control_score_changes_with_composed_bundle(monkeypatch, tmp_path):
    task = _task(tmp_path)
    first_bundle = _bundle(gold_hops=2, issue_hops=2)
    second_bundle = _bundle(gold_hops=0, issue_hops=0)

    def backend_for(bundle: ScoreBundle) -> LocalizationEvaluationResult:
        task_result = LocalizationEvaluationTaskResult(
            task=LCATask(
                identity=task.identity,
                context=task.context,
                gold=LCAGold(changed_files=["a.py"]),
                repo=task.repo,
            ),
            score_bundle=bundle,
            prediction=LocalizationPrediction(predicted_files=["a.py"]),
        )
        summary = summarize_batch_scores({"task": bundle})
        return LocalizationEvaluationResult(
            score_summary=summary,
            machine_score=summary.aggregate_composed_score,
            items=[task_result],
            failure=None,
        )

    monkeypatch.setattr("harness.orchestration.evaluation.evaluate_localization_batch", lambda **kwargs: backend_for(first_bundle))
    res_first = evaluate_policy_on_item(task, None, None, 0)

    monkeypatch.setattr("harness.orchestration.evaluation.evaluate_localization_batch", lambda **kwargs: backend_for(second_bundle))
    res_second = evaluate_policy_on_item(task, None, None, 0)

    assert res_first.control_score == pytest.approx(first_bundle.composed_score)
    assert res_second.control_score == pytest.approx(second_bundle.composed_score)
    assert res_second.control_score > res_first.control_score
