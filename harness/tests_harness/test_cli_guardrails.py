from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

import run as run_module
from harness.entrypoints.models.requests import HostedLocalizationBaselineRequest
from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.telemetry.hosted.baselines import baseline_cache_path
from harness.telemetry.tracing.session_policy import SessionConfig, resolve_session_id


def _task(repo_owner: str, repo_name: str, base_sha: str, dataset: str = "d", config: str = "c", split: str = "s") -> LCATask:
    identity = LCATaskIdentity(
        dataset_name=dataset,
        dataset_config=config,
        dataset_split=split,
        repo_owner=repo_owner,
        repo_name=repo_name,
        base_sha=base_sha,
    )
    ctx = LCAContext(issue_title="t", issue_body="b")
    gold = LCAGold(changed_files=["a.py"])
    return LCATask(identity=identity, context=ctx, gold=gold)


def _stub_cli_baseline(monkeypatch, tmp_path, run_stub, cache_root=None, baseline_root=None):
    hf_cache_dir = cache_root or tmp_path / "hf_cache"
    baseline_cache_dir = baseline_root or tmp_path / "baseline_cache"
    monkeypatch.setenv("HF_LCA_CACHE_DIR", str(hf_cache_dir))
    monkeypatch.setenv("HARNESS_BASELINE_CACHE_DIR", str(baseline_cache_dir))
    monkeypatch.setattr(run_module.harness_run, "ensure_langfuse_auth", lambda: None)
    monkeypatch.setattr(run_module.harness_run, "flush_langfuse", lambda: None)
    monkeypatch.setattr(
        run_module,
        "fetch_hf_localization_dataset",
        lambda *args, **kwargs: [
            SimpleNamespace(task_id="t1", identity=SimpleNamespace(task_id=lambda: "t1"))
        ],
    )

    def fake_select(tasks, max_items, offset):
        return tasks, {
            "selected_count": len(tasks),
            "offset": offset,
            "limit": max_items,
            "total_available": len(tasks),
            "preview": [getattr(tasks[0], "task_id", None) or tasks[0].identity.task_id()],
        }

    monkeypatch.setattr(run_module, "_select_tasks", fake_select)
    monkeypatch.setattr(
        run_module,
        "_compute_projection",
        lambda *args, **kwargs: {"total": {"planned": 0, "hard_cap": 0}, "localization": {}, "writer": {}},
    )
    monkeypatch.setattr(run_module, "_require_confirmation", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module, "run_hosted_localization_baseline", run_stub)
    return hf_cache_dir, baseline_cache_dir


def test_parse_defaults():
    args = run_module._parse_args(["baseline", "--config", "py", "--split", "dev"])
    assert args.max_items is None
    assert args.offset == 0
    assert args.command == "baseline"
    args_exp = run_module._parse_args(["experiment", "--config", "java", "--split", "test", "--max-items", "5"])
    assert args_exp.command == "experiment"
    assert args_exp.max_items == 5


def test_removed_flags_rejected():
    with pytest.raises(SystemExit):
        run_module._parse_args(["baseline", "--config", "py", "--split", "dev", "--dataset", "foo"])
    with pytest.raises(SystemExit):
        run_module._parse_args(["baseline", "--config", "py", "--split", "dev", "--mode", "localization-baseline"])
    with pytest.raises(SystemExit):
        run_module._parse_args(["experiment", "--config", "py", "--split", "dev", "--assume-input-tokens", "10"])
    with pytest.raises(SystemExit):
        run_module._parse_args(["experiment", "--config", "py", "--split", "dev", "--dataset-source", "huggingface"])


def test_select_tasks_is_deterministic_and_offsets():
    tasks = [
        _task("o2", "r2", "b2"),
        _task("o1", "r1", "b1"),
    ]
    selected, info = run_module._select_tasks(tasks, max_items=1, offset=0)
    assert [t.identity.task_id() for t in selected] == [_task("o1", "r1", "b1").identity.task_id()]
    assert info["selected_count"] == 1
    selected_offset, info_offset = run_module._select_tasks(tasks, max_items=1, offset=1)
    assert [t.identity.task_id() for t in selected_offset] == [_task("o2", "r2", "b2").identity.task_id()]
    assert info_offset["offset"] == 1


def test_projection_fail_closed_on_unknown_pricing_without_override(monkeypatch):
    monkeypatch.setattr(run_module, "cost_details_for_usage", lambda *args, **kwargs: None)
    with pytest.raises(ValueError):
        run_module._compute_projection(1)


def test_projection_with_override(monkeypatch):
    # Override pricing by stubbing cost_details_for_usage
    def fake_cost(model, usage):
        # scale so hard-cap totals differ from planned
        return {"total": (usage.get("input", 0) + usage.get("output", 0)) * 0.000001}

    monkeypatch.setattr(run_module, "cost_details_for_usage", fake_cost)
    proj = run_module._compute_projection(3)
    assert proj["total"]["planned"] < proj["total"]["hard_cap"]
    assert proj["localization"]["planned"]["cost"] < proj["localization"]["hard_cap"]["cost"]


def test_confirmation_fallback_line_prompt(monkeypatch):
    # Force non-raw path and force TTY true
    monkeypatch.setattr(run_module, "termios", None)
    monkeypatch.setattr(run_module, "tty", None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "space")
    run_module._require_confirmation(skip_confirmation=False, selection={}, projection={})


def test_confirmation_non_interactive_rejected(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(RuntimeError):
        run_module._require_confirmation(skip_confirmation=False, selection={}, projection={})


def test_projection_only_skips_execution(monkeypatch):
    calls = {"baseline": 0}

    def fake_fetch(name, dataset_config=None, dataset_split=None, revision=None):
        return [_task("o", "r", "b")]

    def fake_projection(*args, **kwargs):
        return {"total": {"planned": 0, "hard_cap": 0}, "localization": {}, "writer": {}}

    def fake_confirm(*args, **kwargs):
        return None

    def fake_run(*args, **kwargs):
        calls["baseline"] += 1
        return None

    monkeypatch.setattr(run_module, "fetch_hf_localization_dataset", fake_fetch)
    monkeypatch.setattr(run_module, "_compute_projection", fake_projection)
    monkeypatch.setattr(run_module, "_require_confirmation", fake_confirm)
    monkeypatch.setattr(run_module, "run_hosted_localization_baseline", fake_run)
    argv = [
        "baseline",
        "--config",
        "py",
        "--split",
        "dev",
        "--projection-only",
        "--yes",
        "--max-items",
        "1",
    ]
    run_module.main(argv)
    assert calls["baseline"] == 0, "Projection-only should exit before executing baseline run"


def test_hf_loader_is_used(monkeypatch):
    calls = {"loader": None, "baseline": None}

    def fake_loader(name, dataset_config=None, dataset_split=None, revision=None):
        calls["loader"] = (name, dataset_config, dataset_split, revision)
        return [SimpleNamespace(task_id="t1", identity=SimpleNamespace(task_id=lambda: "t1"))]

    def fake_select(tasks, max_items, offset):
        return tasks, {"selected_count": 1, "offset": offset, "limit": max_items, "total_available": 1, "preview": ["t1"]}

    def fake_projection(count):
        return {"total": {"planned": 0, "hard_cap": 0}, "localization": {}, "writer": {}}

    def fake_confirm(skip_confirmation, selection, projection):
        calls["baseline"] = True

    monkeypatch.setattr(run_module, "fetch_hf_localization_dataset", fake_loader)
    monkeypatch.setattr(run_module, "_select_tasks", fake_select)
    monkeypatch.setattr(run_module, "_compute_projection", fake_projection)
    monkeypatch.setattr(run_module, "_require_confirmation", fake_confirm)
    monkeypatch.setattr(run_module, "run_hosted_localization_baseline", lambda *args, **kwargs: None)

    run_module.main(["baseline", "--config", "py", "--split", "dev", "--max-items", "1", "--yes"])
    assert calls["loader"] == ("JetBrains-Research/lca-bug-localization", "py", "dev", None)


def test_requests_do_not_expose_dataset_source():
    req = HostedLocalizationBaselineRequest(dataset="d", dataset_config="c", dataset_split="s")
    assert "dataset_source" not in req.model_fields


def test_max_workers_validation():
    argv = ["baseline", "--config", "py", "--split", "dev", "--max-items", "1", "--max-workers", "0"]
    with pytest.raises(ValueError):
        run_module.main(argv)


def test_cli_fresh_session_per_invocation(monkeypatch, tmp_path):
    sessions: list[SessionConfig | None] = []

    def fake_run(req, materializer=None, tasks=None):
        sessions.append(req.session)
        return SimpleNamespace(items=[], failure=None)

    _stub_cli_baseline(monkeypatch, tmp_path, fake_run)
    argv = ["baseline", "--config", "py", "--split", "dev", "--max-items", "1", "--yes"]

    run_module.main(argv)
    run_module.main(argv)

    assert len(sessions) == 2
    assert all(isinstance(sess, SessionConfig) for sess in sessions)
    assert sessions[0].session_scope == "run"
    assert sessions[1].session_scope == "run"
    assert sessions[0].session_id and sessions[1].session_id
    assert sessions[0].session_id != sessions[1].session_id
    assert resolve_session_id(sessions[0], run_id="run-id") == sessions[0].session_id


def test_cli_respects_explicit_session_id(monkeypatch, tmp_path):
    sessions: list[SessionConfig | None] = []

    def fake_run(req, materializer=None, tasks=None):
        sessions.append(req.session)
        return SimpleNamespace(items=[], failure=None)

    _stub_cli_baseline(monkeypatch, tmp_path, fake_run)
    argv = [
        "baseline",
        "--config",
        "py",
        "--split",
        "dev",
        "--max-items",
        "1",
        "--yes",
        "--session-id",
        "explicit-session",
    ]

    run_module.main(argv)
    assert len(sessions) == 1
    assert sessions[0].session_id == "explicit-session"
    assert sessions[0].session_scope == "run"


def test_invalidate_baseline_cache_removes_bundle(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(req, materializer=None, tasks=None):
        calls.append(True)
        return SimpleNamespace(items=[], failure=None)

    hf_cache_root = tmp_path / "hf_cache_existing"
    hf_cache_root.mkdir(parents=True, exist_ok=True)
    (hf_cache_root / "marker.txt").write_text("cached")
    _stub_cli_baseline(monkeypatch, tmp_path, fake_run, cache_root=hf_cache_root)
    bundle_path = baseline_cache_path(
        run_module.harness_run.HF_DATASET_NAME,
        dataset_config="py",
        dataset_split="dev",
        dataset_version=None,
    )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text("{}")

    argv = [
        "baseline",
        "--config",
        "py",
        "--split",
        "dev",
        "--max-items",
        "1",
        "--yes",
        "--invalidate-baseline-cache",
    ]

    run_module.main(argv)
    out = capsys.readouterr().out

    assert "[CACHE] Baseline cache invalidated" in out
    assert calls == [True]
    assert not bundle_path.exists()
    assert (hf_cache_root / "marker.txt").exists(), "HF cache should remain untouched"


def test_invalidate_baseline_cache_missing_is_noop(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(req, materializer=None, tasks=None):
        calls.append(True)
        return SimpleNamespace(items=[], failure=None)

    hf_cache_root = tmp_path / "hf_cache_missing"
    hf_cache_root.mkdir(parents=True, exist_ok=True)
    (hf_cache_root / "marker.txt").write_text("cached")
    _stub_cli_baseline(monkeypatch, tmp_path, fake_run, cache_root=hf_cache_root)
    bundle_path = baseline_cache_path(
        run_module.harness_run.HF_DATASET_NAME,
        dataset_config="py",
        dataset_split="dev",
        dataset_version=None,
    )

    argv = [
        "baseline",
        "--config",
        "py",
        "--split",
        "dev",
        "--max-items",
        "1",
        "--yes",
        "--invalidate-baseline-cache",
    ]

    run_module.main(argv)
    out = capsys.readouterr().out

    assert "Baseline cache not found" in out
    assert calls == [True]
    assert not bundle_path.exists()
    assert (hf_cache_root / "marker.txt").exists(), "HF cache should remain untouched"
