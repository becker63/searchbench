from __future__ import annotations

from pathlib import Path

from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory
from harness.localization.models import LCAContext, LCAGold, LocalizationPrediction, LCATask, LCATaskIdentity
from harness.localization.runtime.evaluate import (
    LocalizationEvaluationTaskResult,
    evaluate_localization_batch,
)
from harness.orchestration.evaluation import evaluate_policy_on_item
from harness.scoring import ScoreBundle, ScoreContext, ScoreEngine


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


def _bundle(gold_hops: int = 0, issue_hops: int = 0) -> ScoreBundle:
    return ScoreEngine().evaluate(
        ScoreContext(
            predicted_files=("a.py",),
            gold_files=("a.py",),
            gold_min_hops=gold_hops,
            issue_min_hops=issue_hops,
            per_prediction_hops={"a.py": gold_hops},
        )
    )


def test_evaluation_backend_success(monkeypatch, tmp_path) -> None:
    task = _make_task(tmp_path)
    bundle = _bundle()

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.run_localization_task",
        lambda *a, **k: (LocalizationPrediction(predicted_files=["a.py"]), bundle, None, None),
        raising=False,
    )
    result = evaluate_localization_batch(
        tasks=[task], dataset_source="test", materializer=None, parent_trace=None
    )
    assert result.failure is None
    assert result.score_summary is not None
    assert result.score_summary.aggregate_composed_score == 1.0
    assert result.machine_score == result.score_summary.aggregate_composed_score
    assert result.items and result.items[0].score_bundle.composed_score == 1.0
    assert (
        LocalizationEvaluationTaskResult.model_fields["prediction"].annotation
        is LocalizationPrediction
    )
    assert isinstance(result.items[0].prediction, LocalizationPrediction)
    assert result.items[0].prediction.predicted_files == ["a.py"]


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


def test_orchestration_evaluation_uses_shared_backend(monkeypatch, tmp_path) -> None:
    bundle = _bundle()
    task = _make_task(tmp_path)

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.run_localization_task",
        lambda *a, **k: (LocalizationPrediction(predicted_files=["a.py"]), bundle, None, None),
        raising=False,
    )
    monkeypatch.setattr("harness.orchestration.evaluation._read_policy", lambda: "policy", raising=False)
    eval_result = evaluate_policy_on_item(task, None, None, 0)
    assert eval_result.metrics.get("score") == 1.0
    assert eval_result.ic_result.get("predicted_files") == ["a.py"]
    assert eval_result.control_score == 1.0


def test_backend_accepts_hf_dataset_source(monkeypatch, tmp_path) -> None:
    task = _make_task(tmp_path)
    bundle = _bundle()

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.run_localization_task",
        lambda *a, **k: (LocalizationPrediction(predicted_files=["a.py"]), bundle, None, None),
        raising=False,
    )
    result = evaluate_localization_batch(
        tasks=[task], dataset_source="huggingface", materializer=None, parent_trace=None
    )
    assert result.failure is None
    assert result.machine_score == 1.0


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


def test_score_summary_failure_is_typed(monkeypatch, tmp_path) -> None:
    task = _make_task(tmp_path)
    bundle = _bundle()

    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.run_localization_task",
        lambda *a, **k: (LocalizationPrediction(predicted_files=["a.py"]), bundle, None, None),
        raising=False,
    )
    monkeypatch.setattr(
        "harness.localization.runtime.evaluate.summarize_batch_scores",
        lambda *_a, **_k: (_ for _ in ()).throw(LocalizationEvaluationError(LocalizationFailureCategory.SCORING, "bad summary")),
        raising=True,
    )
    result = evaluate_localization_batch(
        tasks=[task], dataset_source="test", materializer=None, parent_trace=None
    )
    assert result.failure is not None
    assert result.failure.category == LocalizationFailureCategory.SCORING
