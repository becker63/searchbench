from __future__ import annotations

import pytest

from harness.localization.evaluate import LocalizationEvaluationResult, LocalizationEvaluationTaskResult, MachineScorePolicy
from harness.localization.models import LocalizationMetrics, LCAContext, LCATaskIdentity, LCAGold, LCATask
from harness.orchestration.runtime import evaluate_policy_on_item
from harness.orchestration.types import ICTaskPayload


def test_machine_uses_machine_score(monkeypatch, tmp_path):
    ic_task = ICTaskPayload(
        identity=LCATaskIdentity(
            dataset_name="lca",
            dataset_config="py",
            dataset_split="dev",
            repo_owner="o",
            repo_name="r",
            base_sha="abc",
        ),
        context=LCAContext(issue_title="bug", issue_body="body"),
        repo=str(tmp_path),
        changed_files=["a.py"],
    )
    metrics = LocalizationMetrics(precision=0.1, recall=0.1, f1=0.1, hit=0.0, score=0.1)
    task_result = LocalizationEvaluationTaskResult(
        task=LCATask(
            identity=ic_task.identity,
            context=ic_task.context,
            gold=LCAGold(changed_files=["a.py"]),
            repo=ic_task.repo,
        ),
        metrics=metrics,
        prediction=["a.py"],
    )
    eval_result = LocalizationEvaluationResult(
        aggregate_score=0.1,
        aggregate_metrics=metrics,
        machine_score=0.5,
        items=[task_result],
        failure=None,
    )

    def fake_backend(req):
        return eval_result

    monkeypatch.setattr("harness.orchestration.runtime.evaluate_localization_batch", fake_backend)
    result = evaluate_policy_on_item(ic_task, None, None, 0)
    assert result.control_score == 0.5
    assert result.metrics.get("score") == 0.1


def test_machine_score_policy_changes_control(monkeypatch, tmp_path):
    ic_task = ICTaskPayload(
        identity=LCATaskIdentity(
            dataset_name="lca",
            dataset_config="py",
            dataset_split="dev",
            repo_owner="o",
            repo_name="r",
            base_sha="abc",
        ),
        context=LCAContext(issue_title="bug", issue_body="body"),
        repo=str(tmp_path),
        changed_files=["a.py"],
    )
    metrics = LocalizationMetrics(precision=0.2, recall=0.8, f1=0.32, hit=1.0, score=0.32)

    def fake_backend(req):
        return LocalizationEvaluationResult(
            aggregate_score=metrics.score,
            aggregate_metrics=metrics,
            machine_score=metrics.hit if req.machine_score_policy == MachineScorePolicy.HIT else metrics.score,
            items=[
                LocalizationEvaluationTaskResult(
                    task=LCATask(identity=ic_task.identity, context=ic_task.context, gold=LCAGold(changed_files=["a.py"]), repo=ic_task.repo),
                    metrics=metrics,
                    prediction=["a.py"],
                )
            ],
            failure=None,
        )

    monkeypatch.setattr("harness.orchestration.runtime.evaluate_localization_batch", fake_backend)

    # Default policy uses aggregate (score)
    res_default = evaluate_policy_on_item(ic_task, None, None, 0)
    assert res_default.control_score == pytest.approx(metrics.score)
    # Force HIT policy
    from harness.localization.evaluate import LocalizationEvaluationRequest

    monkeypatch.setattr("harness.orchestration.runtime.LocalizationEvaluationRequest", lambda **kwargs: LocalizationEvaluationRequest(machine_score_policy=MachineScorePolicy.HIT, **kwargs))
    res_hit = evaluate_policy_on_item(ic_task, None, None, 0)
    assert res_hit.control_score == pytest.approx(metrics.hit)
    assert res_hit.metrics.get("score") == pytest.approx(metrics.score)
