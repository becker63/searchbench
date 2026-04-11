from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

import run as run_module
from harness.entrypoints.requests import HostedLocalizationBaselineRequest
from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity


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
