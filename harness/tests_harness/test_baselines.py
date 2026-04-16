from __future__ import annotations

import json

from contextlib import contextmanager

from harness.localization.models import (
    LCAContext,
    LCAGold,
    LCATask,
    LCATaskIdentity,
    LocalizationGold,
    LocalizationPrediction,
)
from harness.localization.runtime.evaluate import LocalizationEvaluationResult, LocalizationEvaluationTaskResult
from harness.scoring import ScoreContext, ScoreEngine, summarize_batch_scores, summarize_task_score
from harness.telemetry.hosted import baselines
from harness.telemetry.hosted import experiments
from harness.entrypoints.models.requests import HostedLocalizationBaselineRequest


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

    monkeypatch.setattr("harness.telemetry.tracing.get_langfuse_client", lambda: DummyClient())


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


def _score_bundle(score_hops: int = 0):
    return ScoreEngine().evaluate(
        ScoreContext(
            predicted_files=("a.py",),
            gold_files=("a.py",),
            gold_min_hops=score_hops,
            issue_min_hops=score_hops,
            per_prediction_hops={"a.py": score_hops},
        )
    )


def _eval_result(task: LCATask) -> LocalizationEvaluationResult:
    bundle = _score_bundle()
    summary = summarize_batch_scores({task.task_id: bundle})
    return LocalizationEvaluationResult(
        score_summary=summary,
        machine_score=summary.aggregate_composed_score,
        items=[
            LocalizationEvaluationTaskResult(
                task=task,
                score_bundle=bundle,
                prediction=["a.py"],
                evidence=None,
                materialization=None,
                trace_id=None,
            )
        ],
        failure=None,
    )


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

    def fake_backend(**kwargs):
        calls.append(kwargs)
        return _eval_result(task)

    monkeypatch.setattr(baselines, "evaluate_localization_batch", fake_backend)
    snap = baselines.compute_baseline_for_task(task, dataset_version="v1", runner=None)
    assert calls
    assert snap.score_summary.composed_score == 1.0
    assert snap.prediction.predicted_files == ["a.py"]


def test_baseline_dataset_span_gets_aggregates(monkeypatch):
    task = _task()
    snap_one = _snapshot(task, version="v1")
    snap_two = _snapshot(task, version="v1", score_hops=1)
    emitted: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []

    class DummySpan:
        def __init__(self):
            self.id = "dataset-span"

        def score_trace(self, **kwargs):
            emitted.append(kwargs)

        def update(self, **kwargs):
            updates.append(kwargs)

    baselines.emit_baseline_dataset_aggregates(DummySpan(), [snap_one, snap_two])
    names = {item["name"] for item in emitted}
    assert names == {
        "baseline.aggregate_composed_score",
        "baseline.aggregate_gold_hop",
        "baseline.aggregate_issue_hop",
    }
    assert updates
    metadata = updates[-1].get("metadata")
    assert metadata is not None
    assert metadata["selected_count"] == 2
    assert metadata["score_summary"]["aggregate_composed_score"] == 0.75


def _snapshot(task: LCATask, version: str = "v1", score_hops: int = 0) -> baselines.BaselineSnapshot:
    bundle = _score_bundle(score_hops)
    score_summary = summarize_task_score(task.task_id, bundle)
    return baselines.BaselineSnapshot(
        dataset_name=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        dataset_version=version,
        identity=task.task_id,
        prediction=LocalizationPrediction(predicted_files=["a.py"]),
        gold=LocalizationGold(changed_files=["a.py"]),
        score_summary=score_summary,
        repo_owner=task.identity.repo_owner,
        repo_name=task.identity.repo_name,
        base_sha=task.identity.base_sha,
    )


def test_persist_and_load_baseline_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_BASELINE_CACHE_DIR", str(tmp_path))
    task = _task()
    snap = _snapshot(task, version="rev1")
    bundle = baselines.make_baseline_bundle([snap], dataset_name=task.identity.dataset_name, dataset_version="rev1")
    path = baselines.persist_baseline_bundle(
        bundle,
        dataset_name=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        dataset_version="rev1",
    )
    expected = baselines.baseline_cache_path(
        task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        dataset_version="rev1",
        root=tmp_path,
    )
    assert path == expected
    # File is valid JSON (no PosixPath serialization issues)
    json.loads(path.read_text())
    loaded = baselines.load_persisted_baseline_bundle(
        dataset_name=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        dataset_version="rev1",
    )
    assert loaded is not None
    assert loaded.items and loaded.items[0].identity == snap.identity


def test_run_hosted_localization_baseline_uses_cached_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_BASELINE_CACHE_DIR", str(tmp_path))
    task = _task()
    snap = _snapshot(task, version="rev2")
    bundle = baselines.make_baseline_bundle([snap], dataset_name=task.identity.dataset_name, dataset_version="rev2")
    baselines.persist_baseline_bundle(
        bundle,
        dataset_name=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        dataset_version="rev2",
    )

    def _raise_fetch(*args, **kwargs):
        raise AssertionError("fetch should not be called")

    def _raise_compute(*args, **kwargs):
        raise AssertionError("compute should not be called")

    monkeypatch.setattr(experiments, "fetch_hf_localization_dataset", _raise_fetch)
    monkeypatch.setattr(experiments, "compute_baseline_for_task", _raise_compute)

    req = HostedLocalizationBaselineRequest(
        dataset=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        version="rev2",
    )
    result = experiments.run_hosted_localization_baseline(req)
    assert result.items and result.items[0].identity == snap.identity


def test_run_hosted_localization_baseline_persists_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_BASELINE_CACHE_DIR", str(tmp_path))
    task = _task()
    snap = _snapshot(task, version="rev3")
    calls = {"compute": 0, "fetch": 0}

    def fake_fetch(*args, **kwargs):
        calls["fetch"] += 1
        return [task]

    def fake_compute(*args, **kwargs):
        calls["compute"] += 1
        return snap

    monkeypatch.setattr(experiments, "fetch_hf_localization_dataset", fake_fetch)
    monkeypatch.setattr(experiments, "compute_baseline_for_task", fake_compute)
    monkeypatch.setattr(experiments, "flush_langfuse", lambda: None)

    req = HostedLocalizationBaselineRequest(
        dataset=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        version="rev3",
    )
    result = experiments.run_hosted_localization_baseline(req)
    assert calls["fetch"] == 1
    assert calls["compute"] == 1
    assert result.items and result.items[0].identity == snap.identity

    bundle_path = baselines.baseline_cache_path(
        task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        dataset_version="rev3",
        root=tmp_path,
    )
    assert bundle_path.exists()
    loaded = baselines.load_persisted_baseline_bundle(
        dataset_name=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        dataset_version="rev3",
    )
    assert loaded is not None
    assert loaded.items and loaded.items[0].identity == snap.identity


def test_cached_baseline_emits_root_scores(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_BASELINE_CACHE_DIR", str(tmp_path))
    task = _task()
    snap = _snapshot(task, version="rev-hit")
    bundle = baselines.make_baseline_bundle([snap], dataset_name=task.identity.dataset_name, dataset_version="rev-hit")
    baselines.persist_baseline_bundle(
        bundle,
        dataset_name=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        dataset_version="rev-hit",
    )

    captured_scores: list[dict[str, object]] = []
    metadata_updates: list[dict[str, object]] = []

    class DummySpan:
        def __init__(self):
            self.id = "dataset-span"
            self.trace_id = "trace-id"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, **kwargs):
            metadata_updates.append(kwargs)

        def score(self, **kwargs):
            raise AssertionError("dataset aggregates should use trace-level scores")

        def score_trace(self, **kwargs):
            captured_scores.append(kwargs)

    @contextmanager
    def span_cm(*_args, **_kwargs):
        yield DummySpan()

    monkeypatch.setattr(experiments, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(experiments, "flush_langfuse", lambda: None)
    monkeypatch.setattr(
        experiments,
        "fetch_hf_localization_dataset",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetch should not run")),
    )
    req = HostedLocalizationBaselineRequest(
        dataset=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        version="rev-hit",
    )
    result = experiments.run_hosted_localization_baseline(req)
    assert result.items and result.items[0].identity == task.task_id
    names = {item["name"] for item in captured_scores}
    assert names == {
        "baseline.aggregate_composed_score",
        "baseline.aggregate_gold_hop",
        "baseline.aggregate_issue_hop",
    }
    assert metadata_updates
    metadata = metadata_updates[-1].get("metadata")
    assert metadata and metadata.get("cache_hit") is True
    assert metadata.get("selected_count") == 1
    assert metadata["score_summary"]["aggregate_composed_score"] == snap.score_summary.composed_score


def test_fresh_baseline_emits_root_scores(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_BASELINE_CACHE_DIR", str(tmp_path))
    task = _task()
    snap = _snapshot(task, version="rev-fresh")
    captured_scores: list[dict[str, object]] = []
    metadata_updates: list[dict[str, object]] = []

    class DummySpan:
        def __init__(self):
            self.id = "dataset-span"
            self.trace_id = "trace-id"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, **kwargs):
            metadata_updates.append(kwargs)

        def score(self, **kwargs):
            raise AssertionError("dataset aggregates should use trace-level scores")

        def score_trace(self, **kwargs):
            captured_scores.append(kwargs)

    @contextmanager
    def span_cm(*_args, **_kwargs):
        yield DummySpan()

    monkeypatch.setattr(experiments, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(experiments, "flush_langfuse", lambda: None)
    monkeypatch.setattr(
        experiments,
        "fetch_hf_localization_dataset",
        lambda *a, **k: [task],
    )
    monkeypatch.setattr(
        experiments,
        "compute_baseline_for_task",
        lambda *a, **k: snap,
    )
    req = HostedLocalizationBaselineRequest(
        dataset=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        version="rev-fresh",
    )
    result = experiments.run_hosted_localization_baseline(req)
    assert result.items and result.items[0].identity == task.task_id
    names = {item["name"] for item in captured_scores}
    assert names == {
        "baseline.aggregate_composed_score",
        "baseline.aggregate_gold_hop",
        "baseline.aggregate_issue_hop",
    }
    metadata = metadata_updates[-1].get("metadata")
    assert metadata and metadata.get("cache_hit") is False
    assert metadata.get("selected_count") == 1


def test_baseline_cache_path_respects_env_and_dataset(monkeypatch, tmp_path):
    cache_root = tmp_path / "baseline_cache"
    monkeypatch.setenv("HARNESS_BASELINE_CACHE_DIR", str(cache_root))
    path = baselines.baseline_cache_path(
        "JetBrains-Research/lca-bug-localization", dataset_config="py", dataset_split="dev", dataset_version="rev1"
    )
    expected = cache_root / "JetBrains-Research__lca-bug-localization" / "py" / "dev" / "rev1.json"
    assert path == expected
