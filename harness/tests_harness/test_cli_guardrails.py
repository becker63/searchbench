from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

import run as run_module
from harness.entrypoints.models.requests import HostedLocalizationBaselineRequest
from harness.localization.materialization.worktree import RepoMaterializationRequest, WorktreeManager
from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.telemetry.hosted.baselines import baseline_cache_path
from harness.telemetry.tracing.session_policy import SessionConfig, resolve_session_id


LANGFUSE_DATASET = "canonical-lca"


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
    worktree_cache_dir = cache_root or tmp_path / "worktree_cache"
    baseline_cache_dir = baseline_root or tmp_path / "baseline_cache"
    monkeypatch.setenv("HARNESS_WORKTREE_CACHE_DIR", str(worktree_cache_dir))
    monkeypatch.setenv("HARNESS_BASELINE_CACHE_DIR", str(baseline_cache_dir))
    monkeypatch.setattr(run_module.harness_run, "ensure_langfuse_auth", lambda: None)
    monkeypatch.setattr(run_module.harness_run, "flush_langfuse", lambda: None)
    monkeypatch.setattr(
        run_module,
        "fetch_localization_dataset",
        lambda *args, **kwargs: [_task("o", "r", "b", dataset=LANGFUSE_DATASET, config="py", split="dev")],
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
    return worktree_cache_dir, baseline_cache_dir


def _worktree_path(cache_root, *, repo_owner: str = "o", repo_name: str = "r", base_sha: str = "b", config: str = "py"):
    manager = WorktreeManager(cache_root=cache_root)
    request = RepoMaterializationRequest(
        dataset_name=LANGFUSE_DATASET,
        dataset_config=config,
        dataset_split="dev",
        repo_owner=repo_owner,
        repo_name=repo_name,
        base_sha=base_sha,
    )
    return manager._worktree_path(manager._cache_key(request))  # pyright: ignore[reportPrivateUsage]


def test_parse_defaults():
    args = run_module._parse_args(["baseline", "--dataset", LANGFUSE_DATASET])
    assert args.max_items is None
    assert args.offset == 0
    assert args.command == "baseline"
    args_exp = run_module._parse_args(["experiment", "--dataset", LANGFUSE_DATASET, "--max-items", "5"])
    assert args_exp.command == "experiment"
    assert args_exp.max_items == 5


def test_removed_flags_rejected():
    with pytest.raises(SystemExit):
        run_module._parse_args(["baseline", "--dataset", LANGFUSE_DATASET, "--mode", "localization-baseline"])
    with pytest.raises(SystemExit):
        run_module._parse_args(["experiment", "--dataset", LANGFUSE_DATASET, "--assume-input-tokens", "10"])
    with pytest.raises(SystemExit):
        run_module._parse_args(["experiment", "--dataset", LANGFUSE_DATASET, "--dataset-source", "huggingface"])


def test_hf_flags_are_sync_only():
    with pytest.raises(SystemExit):
        run_module._parse_args(["baseline", "--dataset", LANGFUSE_DATASET, "--config", "py"])
    with pytest.raises(SystemExit):
        run_module._parse_args(["experiment", "--dataset", LANGFUSE_DATASET, "--revision", "rev1"])

    args = run_module._parse_args([
        "sync-hf",
        "--dataset",
        LANGFUSE_DATASET,
        "--config",
        "py",
        "--split",
        "dev",
        "--revision",
        "rev1",
    ])
    assert args.command == "sync-hf"
    assert args.config == "py"
    assert args.split == "dev"
    assert args.revision == "rev1"


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

    def fake_fetch(name, version=None):
        return [_task("o", "r", "b")]

    def fake_projection(*args, **kwargs):
        return {"total": {"planned": 0, "hard_cap": 0}, "localization": {}, "writer": {}}

    def fake_confirm(*args, **kwargs):
        return None

    def fake_run(*args, **kwargs):
        calls["baseline"] += 1
        return None

    monkeypatch.setattr(run_module, "fetch_localization_dataset", fake_fetch)
    monkeypatch.setattr(run_module, "_compute_projection", fake_projection)
    monkeypatch.setattr(run_module, "_require_confirmation", fake_confirm)
    monkeypatch.setattr(run_module, "run_hosted_localization_baseline", fake_run)
    argv = [
        "baseline",
        "--dataset",
        LANGFUSE_DATASET,
        "--projection-only",
        "--yes",
        "--max-items",
        "1",
    ]
    run_module.main(argv)
    assert calls["baseline"] == 0, "Projection-only should exit before executing baseline run"


def test_projection_only_with_invalidation_skips_destructive_cache_removal(monkeypatch, tmp_path):
    calls = {"baseline": 0}

    def fake_fetch(name, version=None):
        return [_task("o", "r", "b", dataset=name, config="py", split="dev")]

    def fake_run(*args, **kwargs):
        calls["baseline"] += 1
        return None

    monkeypatch.setenv("HARNESS_WORKTREE_CACHE_DIR", str(tmp_path / "worktree_cache"))
    monkeypatch.setenv("HARNESS_BASELINE_CACHE_DIR", str(tmp_path / "baseline_cache"))
    monkeypatch.setattr(run_module.harness_run, "ensure_langfuse_auth", lambda: None)
    monkeypatch.setattr(run_module.harness_run, "flush_langfuse", lambda: None)
    monkeypatch.setattr(run_module, "fetch_localization_dataset", fake_fetch)
    monkeypatch.setattr(run_module, "_compute_projection", lambda *args, **kwargs: {"total": {"planned": 0, "hard_cap": 0}, "localization": {}, "writer": {}})
    monkeypatch.setattr(run_module, "_require_confirmation", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module, "run_hosted_localization_baseline", fake_run)
    bundle_path = baseline_cache_path(
        LANGFUSE_DATASET,
        dataset_version=None,
    )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text("{}")
    worktree_path = _worktree_path(tmp_path / "worktree_cache")
    worktree_path.mkdir(parents=True)
    (worktree_path / ".complete").write_text("")

    run_module.main([
        "baseline",
        "--dataset",
        LANGFUSE_DATASET,
        "--projection-only",
        "--yes",
        "--max-items",
        "1",
        "--invalidate-baseline-cache",
    ])

    assert calls["baseline"] == 0
    assert bundle_path.exists()
    assert worktree_path.exists()
    assert (worktree_path / ".complete").exists()


def test_langfuse_loader_is_used(monkeypatch):
    calls = {"loader": None, "baseline": None}

    def fake_loader(name, version=None):
        calls["loader"] = (name, version)
        return [SimpleNamespace(task_id="t1", identity=SimpleNamespace(task_id=lambda: "t1"))]

    def fake_select(tasks, max_items, offset):
        return tasks, {"selected_count": 1, "offset": offset, "limit": max_items, "total_available": 1, "preview": ["t1"]}

    def fake_projection(count):
        return {"total": {"planned": 0, "hard_cap": 0}, "localization": {}, "writer": {}}

    def fake_confirm(skip_confirmation, selection, projection):
        calls["baseline"] = True

    monkeypatch.setattr(run_module, "fetch_localization_dataset", fake_loader)
    monkeypatch.setattr(run_module, "_select_tasks", fake_select)
    monkeypatch.setattr(run_module, "_compute_projection", fake_projection)
    monkeypatch.setattr(run_module, "_require_confirmation", fake_confirm)
    monkeypatch.setattr(run_module, "run_hosted_localization_baseline", lambda *args, **kwargs: None)

    run_module.main(["baseline", "--dataset", LANGFUSE_DATASET, "--version", "v1", "--max-items", "1", "--yes"])
    assert calls["loader"] == (LANGFUSE_DATASET, "v1")


def test_requests_do_not_expose_dataset_provenance():
    req = HostedLocalizationBaselineRequest(dataset="d")
    assert req.dataset == "d"
    assert "dataset_provenance" not in HostedLocalizationBaselineRequest.model_fields


def test_max_workers_validation():
    argv = ["baseline", "--dataset", LANGFUSE_DATASET, "--max-items", "1", "--max-workers", "0"]
    with pytest.raises(ValueError):
        run_module.main(argv)


def test_cli_fresh_session_per_invocation(monkeypatch, tmp_path):
    sessions: list[SessionConfig | None] = []

    def fake_run(req, worktree_manager=None, tasks=None):
        sessions.append(req.session)
        return SimpleNamespace(records=[], failure=None)

    _stub_cli_baseline(monkeypatch, tmp_path, fake_run)
    argv = ["baseline", "--dataset", LANGFUSE_DATASET, "--max-items", "1", "--yes"]

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

    def fake_run(req, worktree_manager=None, tasks=None):
        sessions.append(req.session)
        return SimpleNamespace(records=[], failure=None)

    _stub_cli_baseline(monkeypatch, tmp_path, fake_run)
    argv = [
        "baseline",
        "--dataset",
        LANGFUSE_DATASET,
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

    def fake_run(req, worktree_manager=None, tasks=None):
        calls.append(True)
        return SimpleNamespace(records=[], failure=None)

    checkout_cache_root = tmp_path / "checkout_cache_existing"
    checkout_cache_root.mkdir(parents=True, exist_ok=True)
    (checkout_cache_root / "marker.txt").write_text("cached")
    worktree_path = _worktree_path(checkout_cache_root)
    worktree_path.mkdir(parents=True)
    (worktree_path / ".complete").write_text("")
    _stub_cli_baseline(monkeypatch, tmp_path, fake_run, cache_root=checkout_cache_root)
    bundle_path = baseline_cache_path(
        LANGFUSE_DATASET,
        dataset_version=None,
    )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text("{}")

    argv = [
        "baseline",
        "--dataset",
        LANGFUSE_DATASET,
        "--max-items",
        "1",
        "--yes",
        "--invalidate-baseline-cache",
    ]

    run_module.main(argv)
    out = capsys.readouterr().out

    assert "[CACHE] Baseline cache invalidated" in out
    assert "[CACHE] Repo worktree cache invalidated for 1 selected task(s)" in out
    assert calls == [True]
    assert not bundle_path.exists()
    assert not worktree_path.exists()
    assert (checkout_cache_root / "marker.txt").exists(), "checkout cache root marker should remain untouched"


def test_invalidate_baseline_cache_missing_is_noop(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(req, worktree_manager=None, tasks=None):
        calls.append(True)
        return SimpleNamespace(records=[], failure=None)

    checkout_cache_root = tmp_path / "checkout_cache_missing"
    checkout_cache_root.mkdir(parents=True, exist_ok=True)
    (checkout_cache_root / "marker.txt").write_text("cached")
    worktree_path = _worktree_path(checkout_cache_root)
    worktree_path.mkdir(parents=True)
    (worktree_path / ".complete").write_text("")
    _stub_cli_baseline(monkeypatch, tmp_path, fake_run, cache_root=checkout_cache_root)
    bundle_path = baseline_cache_path(
        LANGFUSE_DATASET,
        dataset_version=None,
    )

    argv = [
        "baseline",
        "--dataset",
        LANGFUSE_DATASET,
        "--max-items",
        "1",
        "--yes",
        "--invalidate-baseline-cache",
    ]

    run_module.main(argv)
    out = capsys.readouterr().out

    assert "Baseline cache not found" in out
    assert "[CACHE] Repo worktree cache invalidated for 1 selected task(s)" in out
    assert calls == [True]
    assert not bundle_path.exists()
    assert not worktree_path.exists()
    assert (checkout_cache_root / "marker.txt").exists(), "checkout cache root marker should remain untouched"


def test_invalidate_baseline_cache_only_removes_selected_repo_worktree(monkeypatch, tmp_path):
    def fake_run(req, worktree_manager=None, tasks=None):
        return SimpleNamespace(records=[], failure=None)

    selected_task = _task(
        "selected",
        "repo",
        "a" * 40,
        dataset=LANGFUSE_DATASET,
        config="py",
        split="dev",
    )
    other_task = _task(
        "other",
        "repo",
        "b" * 40,
        dataset=LANGFUSE_DATASET,
        config="py",
        split="dev",
    )
    checkout_cache_root = tmp_path / "checkout_cache_targeted"
    selected_path = _worktree_path(
        checkout_cache_root,
        repo_owner="selected",
        repo_name="repo",
        base_sha="a" * 40,
    )
    other_path = _worktree_path(
        checkout_cache_root,
        repo_owner="other",
        repo_name="repo",
        base_sha="b" * 40,
    )
    selected_path.mkdir(parents=True)
    other_path.mkdir(parents=True)
    (selected_path / ".complete").write_text("")
    (other_path / ".complete").write_text("")
    _stub_cli_baseline(monkeypatch, tmp_path, fake_run, cache_root=checkout_cache_root)
    monkeypatch.setattr(
        run_module,
        "fetch_localization_dataset",
        lambda *args, **kwargs: [selected_task, other_task],
    )
    monkeypatch.setattr(
        run_module,
        "_select_tasks",
        lambda tasks, max_items, offset: (
            [selected_task],
            {
                "selected_count": 1,
                "offset": offset,
                "limit": max_items,
                "total_available": 2,
                "preview": [selected_task.task_id],
            },
        ),
    )

    run_module.main([
        "baseline",
        "--dataset",
        LANGFUSE_DATASET,
        "--max-items",
        "1",
        "--yes",
        "--invalidate-baseline-cache",
    ])

    assert not selected_path.exists()
    assert other_path.exists()
    assert (other_path / ".complete").exists()
