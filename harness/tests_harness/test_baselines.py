from __future__ import annotations

from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.telemetry import baselines


def _install_dummy_langfuse(monkeypatch):
    class _CM:
        def __init__(self):
            self.id = "trace-id"
            self.trace_id = "trace-id"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, **kwargs):
            return None

    class DummyClient:
        def start_as_current_observation(self, **kwargs):
            return _CM()

    monkeypatch.setattr("harness.telemetry.langfuse.get_langfuse_client", lambda: DummyClient())


def _task(repo: str | None = None) -> LCATask:
    identity = LCATaskIdentity(
        dataset_name="lca-bug-localization",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="square",
        repo_name="okhttp",
        base_sha="abc123",
    )
    context = LCAContext(issue_title="bug", issue_body="details")
    gold = LCAGold(changed_files=["a.py"])
    return LCATask(identity=identity, context=context, gold=gold, repo=repo)


def test_baseline_key_is_task_id():
    task = _task()
    key = baselines.baseline_key(task)
    assert key == task.task_id


def test_resolve_and_bundle_roundtrip(tmp_path):
    import pytest
    monkeypatch = pytest.MonkeyPatch()
    _install_dummy_langfuse(monkeypatch)
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    task = _task(str(repo_dir))
    snap = baselines.compute_baseline_for_task(
        task,
        dataset_version="v1",
        runner=lambda *_: (["a.py"], {"source": "runner"}),
    )
    bundle = baselines.make_baseline_bundle([snap], dataset_name=task.identity.dataset_name, dataset_version="v1")
    path = tmp_path / "bundle.json"
    baselines.save_baseline_bundle(bundle, path)
    loaded = baselines.load_baseline_bundle(path)
    resolved = baselines.require_baseline(loaded, task)
    assert resolved.identity == task.task_id
    assert resolved.prediction.predicted_files is not None


def test_require_baseline_errors_when_missing():
    bundle = baselines.make_baseline_bundle([], dataset_name="d")
    task = _task()
    try:
        baselines.require_baseline(bundle, task)
    except ValueError as exc:
        assert "task_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing baseline")


def test_baseline_uses_shared_backend(monkeypatch, tmp_path):
    _install_dummy_langfuse(monkeypatch)
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    task = _task(str(repo_dir))
    calls = []

    from harness.localization.evaluate import LocalizationEvaluationResult, LocalizationEvaluationTaskResult
    from harness.localization.models import LocalizationMetrics, LocalizationPrediction

    def fake_backend(**kwargs):
        calls.append(kwargs)
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

    monkeypatch.setattr(baselines, "evaluate_localization_batch", fake_backend)
    snap = baselines.compute_baseline_for_task(task, dataset_version="v1", runner=None)
    assert calls
    assert snap.metrics.score == 1.0
    assert snap.prediction.predicted_files == ["a.py"]
