from __future__ import annotations

from harness.observability import baselines


def test_baseline_key_uses_id_or_hash():
    item = baselines.normalize_dataset_item({"repo": "r", "symbol": "s", "id": "123"})  # type: ignore[arg-type]
    assert baselines.baseline_key(item) == "123"
    item_no_id = baselines.normalize_dataset_item({"repo": "r", "symbol": "s"})  # type: ignore[arg-type]
    key = baselines.baseline_key(item_no_id)
    assert key and isinstance(key, str)


def test_resolve_baseline_prefers_id():
    item = baselines.normalize_dataset_item({"repo": "r", "symbol": "s", "id": "abc"})  # type: ignore[arg-type]
    snap = baselines.BaselineSnapshot(dataset_name=None, dataset_version=None, item_id="abc", repo="r", symbol="s", jc_result={}, jc_metrics={}, experiment_run_id=None, trace_id=None, metadata={})
    resolved = baselines.resolve_baseline([snap], item)
    assert resolved is snap


def test_baseline_bundle_roundtrip(tmp_path):
    snap = baselines.BaselineSnapshot(
        dataset_name="d",
        dataset_version="v",
        item_id="abc",
        repo="r",
        symbol="s",
        jc_result={"x": 1},
        jc_metrics={"score": 1.0},
        experiment_run_id="run1",
        trace_id="trace1",
        metadata={},
    )
    bundle = baselines.make_baseline_bundle([snap], dataset_name="d", dataset_version="v", baseline_run_id="run1")
    path = tmp_path / "bundle.json"
    baselines.save_baseline_bundle(bundle, path)
    loaded = baselines.load_baseline_bundle(path)
    resolved = baselines.require_baseline(loaded, baselines.normalize_dataset_item({"repo": "r", "symbol": "s"}))  # type: ignore[arg-type]
    metrics = resolved.get("jc_metrics", {})
    assert metrics.get("score") == 1.0


def test_require_baseline_errors_when_missing():
    bundle = baselines.make_baseline_bundle([], dataset_name="d")
    item = baselines.normalize_dataset_item({"repo": "r", "symbol": "s"})  # type: ignore[arg-type]
    try:
        baselines.require_baseline(bundle, item)
    except ValueError as exc:
        assert "repo" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing baseline")
