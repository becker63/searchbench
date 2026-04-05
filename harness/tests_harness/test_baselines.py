from __future__ import annotations

from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.observability import baselines


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
