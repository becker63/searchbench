from __future__ import annotations

from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.observability import experiments


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


def test_experiment_uses_shared_backend(monkeypatch, tmp_path) -> None:
    task = _task(str(tmp_path / "repo"))
    calls = []

    from harness.localization.evaluation_backend import LocalizationEvaluationResult, LocalizationEvaluationTaskResult
    from harness.localization.models import LocalizationMetrics

    def fake_backend(req):
        calls.append(req)
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
