from __future__ import annotations

from pathlib import Path

from harness.localization.runtime.evaluate import (
    DEFAULT_LOCALIZATION_RUNNER,
    MachineScorePolicy,
    evaluate_localization_batch,
)
from harness.localization.errors import LocalizationFailureCategory, LocalizationEvaluationError
from harness.localization.models import LCAContext, LCATask, LCATaskIdentity, LCAGold, LCAPrediction, LocalizationMetrics
from harness.orchestration.runtime import evaluate_policy_on_item
from harness.orchestration.types import ICTaskPayload


def _make_task(tmp_path: Path) -> LCATask:
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="owner",
        repo_name="repo",
        base_sha="abc",
    )
    context = LCAContext(issue_title="bug", issue_body="body")
    gold = LCAGold(changed_files=["a.py"])
    return LCATask(identity=identity, context=context, gold=gold, repo=str(tmp_path))


def test_evaluation_backend_success(monkeypatch, tmp_path) -> None:
    task = _make_task(tmp_path)

    def stub_runner(lca_task: LCATask, repo_path: str, parent: object | None):
        return ["a.py"], {}

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.DEFAULT_LOCALIZATION_RUNNER",
        stub_runner,
        raising=False,
    )
    result = evaluate_localization_batch(
        tasks=[task], dataset_source="test", materializer=None, parent_trace=None
    )
    assert result.failure is None
    assert result.aggregate_score == 1.0
    assert result.items and result.items[0].metrics.score == 1.0


def test_evaluation_backend_failure_maps_category(monkeypatch, tmp_path) -> None:
    task = _make_task(tmp_path)

    def failing_runner(lca_task: LCATask, repo_path: str, parent: object | None):
        raise RuntimeError("runner failed")

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.DEFAULT_LOCALIZATION_RUNNER",
        failing_runner,
        raising=False,
    )
    result = evaluate_localization_batch(
        tasks=[task], dataset_source="test", materializer=None, parent_trace=None
    )
    assert result.failure is not None
    assert result.failure.category == LocalizationFailureCategory.RUNNER


def test_machine_adapter_uses_shared_backend(monkeypatch, tmp_path) -> None:
    def stub_runner(lca_task: LCATask, repo_path: str, parent: object | None):
        return ["a.py"], {}

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.DEFAULT_LOCALIZATION_RUNNER",
        stub_runner,
        raising=False,
    )
    monkeypatch.setattr("harness.orchestration.runtime._read_policy", lambda: "policy", raising=False)
    ic_task = ICTaskPayload(
        identity=_make_task(tmp_path).identity,
        context=_make_task(tmp_path).context,
        repo=str(tmp_path),
        changed_files=["a.py"],
    )
    eval_result = evaluate_policy_on_item(ic_task, None, None, 0)
    assert eval_result.metrics.get("score") == 1.0
    assert eval_result.ic_result.get("predicted_files") == ["a.py"]
    assert eval_result.control_score == 1.0


def test_backend_accepts_hf_dataset_source(monkeypatch, tmp_path) -> None:
    task = _make_task(tmp_path)

    def stub_runner(lca_task: LCATask, repo_path: str, parent: object | None):
        return ["a.py"], {}

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.DEFAULT_LOCALIZATION_RUNNER",
        stub_runner,
        raising=False,
    )
    result = evaluate_localization_batch(
        tasks=[task], dataset_source="huggingface", materializer=None, parent_trace=None
    )
    assert result.failure is None
    assert result.aggregate_score == 1.0


def test_machine_score_policy(monkeypatch, tmp_path) -> None:
    task = _make_task(tmp_path)

    def stub_runner(lca_task: LCATask, repo_path: str, parent: object | None):
        return ["a.py"], {}

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.run_localization_task",
        lambda *a, **k: (
            LCAPrediction(predicted_files=["a.py"]),
            LocalizationMetrics(precision=0.5, recall=0.5, f1=0.5, hit=1.0, score=0.5),
            None,
            None,
        ),
        raising=False,
    )
    result = evaluate_localization_batch(
        tasks=[task],
        dataset_source="test",
        materializer=None,
        parent_trace=None,
        machine_score_policy=MachineScorePolicy.HIT,
    )
    assert result.failure is None
    assert result.machine_score == 1.0
    assert result.items[0].metrics.score == 0.5


def test_typed_failure_propagates(monkeypatch, tmp_path) -> None:
    task = _make_task(tmp_path)

    def failing_runner(lca_task: LCATask, repo_path: str, parent: object | None):
        raise LocalizationEvaluationError(LocalizationFailureCategory.SCORING, "scoring failed")

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.DEFAULT_LOCALIZATION_RUNNER",
        failing_runner,
        raising=False,
    )
    result = evaluate_localization_batch(
        tasks=[task], dataset_source="test", materializer=None, parent_trace=None
    )
    assert result.failure is not None
    assert result.failure.category == LocalizationFailureCategory.SCORING


def test_machine_score_policy_failure_is_typed(monkeypatch, tmp_path) -> None:
    task = _make_task(tmp_path)

    def stub_runner(lca_task: LCATask, repo_path: str, parent: object | None):
        return ["a.py"], {}

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.DEFAULT_LOCALIZATION_RUNNER",
        stub_runner,
        raising=False,
    )
    monkeypatch.setattr(
        "harness.localization.runtime.evaluate._derive_machine_score",
        lambda **kwargs: (_ for _ in ()).throw(LocalizationEvaluationError(LocalizationFailureCategory.SCORING, "bad policy")),
        raising=True,
    )
    result = evaluate_localization_batch(
        tasks=[task], dataset_source="test", materializer=None, parent_trace=None
    )
    assert result.failure is not None
    assert result.failure.category == LocalizationFailureCategory.SCORING


def test_aggregate_metrics_failure_is_typed(monkeypatch, tmp_path) -> None:
    task = _make_task(tmp_path)

    class BadMetrics:
        def __init__(self, *args, **kwargs):
            raise ValueError("aggregate failed")

    def stub_runner(lca_task: LCATask, repo_path: str, parent: object | None):
        return ["a.py"], {}

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.DEFAULT_LOCALIZATION_RUNNER",
        stub_runner,
        raising=False,
    )
    monkeypatch.setattr("harness.localization.runtime.evaluate.LocalizationMetrics", BadMetrics, raising=False)
    result = evaluate_localization_batch(
        tasks=[task], dataset_source="test", materializer=None, parent_trace=None
    )
    assert result.failure is not None
    assert result.failure.category == LocalizationFailureCategory.SCORING
