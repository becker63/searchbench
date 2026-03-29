from __future__ import annotations

from typing import Any, cast

from harness.observability import datasets, experiments
from harness.observability.baselines import BaselineBundle


def test_fetch_dataset_items_from_client(monkeypatch):
    class FakeDataset:
        def __init__(self):
            self._items = [{"repo": "r1", "symbol": "s1", "metadata": {"k": 1}}]

        def items(self):
            return self._items

    class FakeClient:
        def get_dataset(self, name: str, version: str | None = None):
            return FakeDataset()

    monkeypatch.setattr(datasets, "get_langfuse_client", lambda: FakeClient())
    items = datasets.fetch_dataset_items("fake")
    assert items
    first_item = cast(datasets.DatasetItem, items[0])
    assert first_item.get("repo") == "r1"


def test_run_hosted_jc_baseline_experiment(monkeypatch):
    items = [datasets.normalize_dataset_item({"repo": "r", "symbol": "s"})]
    monkeypatch.setattr(experiments, "fetch_dataset_items", lambda name, version=None: items)
    monkeypatch.setattr(experiments, "record_score", lambda *a, **k: None)
    monkeypatch.setattr(experiments, "flush_langfuse", lambda: None)
    monkeypatch.setattr(experiments, "compute_baseline_for_item", lambda *a, **k: {"repo": "r", "symbol": "s", "jc_result": {}, "jc_metrics": {}})

    def fake_runner(dataset_name, dataset_version, experiment_name, task_fn, **kwargs):
        outputs = []
        for it in items:
            outputs.append({"item": it, "output": task_fn(it, None), "trace_id": "t1"})
        return {"run_id": "run1", "items": outputs, "dataset_name": dataset_name, "dataset_version": dataset_version}

    monkeypatch.setattr(experiments, "run_hosted_dataset_experiment", fake_runner)
    bundle_any = cast(Any, experiments.run_hosted_jc_baseline_experiment("fake", version=None))
    assert bundle_any
    first = bundle_any["items"][0]
    assert first["symbol"] == "s"
    assert bundle_any.get("baseline_run_id") == "run1"


def test_run_hosted_ic_optimization_experiment(monkeypatch):
    items = [datasets.normalize_dataset_item({"repo": "r", "symbol": "s"})]
    monkeypatch.setattr(experiments, "fetch_dataset_items", lambda name, version=None: items)
    monkeypatch.setattr(experiments, "record_score", lambda *a, **k: None)
    monkeypatch.setattr(experiments, "flush_langfuse", lambda: None)
    monkeypatch.setattr(experiments, "run_loop", lambda *a, **k: [{"score": 2.0}])
    monkeypatch.setattr(experiments, "resolve_repo_target", lambda ref: "resolved-path")

    def fake_runner(dataset_name, dataset_version, experiment_name, task_fn, **kwargs):
        outputs = []
        for it in items:
            outputs.append({"item": it, "output": task_fn(it, None), "trace_id": "t2"})
        return {"run_id": "run2", "items": outputs, "dataset_name": dataset_name, "dataset_version": dataset_version}

    monkeypatch.setattr(experiments, "run_hosted_dataset_experiment", fake_runner)
    results_any = cast(Any, experiments.run_hosted_ic_optimization_experiment("fake", iterations=1, baselines={"items": [{"repo": "r", "symbol": "s", "jc_result": {}, "jc_metrics": {}}]}))
    assert results_any
    first = results_any[0]
    assert first["item"]["symbol"] == "s"


def test_run_ic_optimization_requires_baseline(monkeypatch):
    items = [datasets.normalize_dataset_item({"repo": "r", "symbol": "s"})]
    monkeypatch.setattr(experiments, "fetch_dataset_items", lambda name, version=None: items)
    try:
        experiments.run_hosted_ic_optimization_experiment("fake", iterations=1, recompute_baselines=False)
    except ValueError as exc:
        assert "Baselines bundle" in str(exc)
    else:
        raise AssertionError("Expected missing baseline error")


def test_run_ic_optimization_uses_provided_baseline(monkeypatch):
    items = [datasets.normalize_dataset_item({"repo": "r", "symbol": "s"})]
    monkeypatch.setattr(experiments, "fetch_dataset_items", lambda name, version=None: items)
    called = {"baseline": False}

    def fail_run_baseline(*args, **kwargs):
        called["baseline"] = True
        raise AssertionError("Should not recompute baseline")

    monkeypatch.setattr(experiments, "run_hosted_jc_baseline_experiment", fail_run_baseline)
    monkeypatch.setattr(experiments, "record_score", lambda *a, **k: None)
    monkeypatch.setattr(experiments, "flush_langfuse", lambda: None)
    monkeypatch.setattr(experiments, "run_loop", lambda *a, **k: [{"score": 1.0}])
    monkeypatch.setattr(experiments, "resolve_repo_target", lambda ref: "/tmp/resolved-r")

    def fake_runner(dataset_name, dataset_version, experiment_name, task_fn, **kwargs):
        outputs = []
        for it in items:
            outputs.append({"item": it, "output": task_fn(it, None), "trace_id": "t3"})
        return {"run_id": "run3", "items": outputs, "dataset_name": dataset_name, "dataset_version": dataset_version}

    monkeypatch.setattr(experiments, "run_hosted_dataset_experiment", fake_runner)
    baselines_bundle: BaselineBundle = {"items": [{"repo": "r", "symbol": "s", "jc_result": {}, "jc_metrics": {}}]}
    results_any = experiments.run_hosted_ic_optimization_experiment("fake", iterations=1, baselines=baselines_bundle)
    assert results_any and not called["baseline"]
