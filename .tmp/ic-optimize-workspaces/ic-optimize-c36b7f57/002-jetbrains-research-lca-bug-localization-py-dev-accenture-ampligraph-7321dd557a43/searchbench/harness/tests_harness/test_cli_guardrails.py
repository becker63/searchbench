from __future__ import annotations

import builtins
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import run as run_module
from harness.entrypoints.flows import (
    FLOW_NARRATIVES,
    IC_OPTIMIZE_NARRATIVE,
    FlowNarrative,
)
from harness.localization.materialization.worktree import (
    RepoMaterializationRequest,
    WorktreeManager,
)
from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.localization.runtime.evaluate import LocalizationEvaluationResult
from harness.telemetry.hosted.hf_lca import HFDatasetSyncResult
from harness.telemetry.observability_models import (
    RunSummary,
    TaskRunSummary,
    TokenUsageBreakdown,
)


LANGFUSE_DATASET = "canonical-lca"
RUNNER = CliRunner()


def _task_summary(backend: str = "ic", status: str = "success") -> TaskRunSummary:
    return TaskRunSummary(
        backend=backend,
        identity="task-id",
        repo="/repo",
        status=status,
        stage=None if status == "success" else "backend_init",
        message=None if status == "success" else "boom",
        step_count=1,
        generation_count=2,
        tool_calls_count=3,
        successful_tool_calls_count=2,
        failed_tool_calls_count=1,
        predicted_files_count=4 if status == "success" else 0,
        duration_seconds=1.5,
        tokens=TokenUsageBreakdown(
            fresh_input_tokens=10,
            cached_input_tokens=20,
            total_input_tokens=30,
            output_tokens=5,
            total_tokens=35,
        ),
        evaluation={"gold_overlap_count": 1.0} if status == "success" else {},
    )


def _task(
    repo_owner: str = "o",
    repo_name: str = "r",
    base_sha: str = "b",
    dataset: str = LANGFUSE_DATASET,
    config: str = "py",
    split: str = "dev",
) -> LCATask:
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


def _worktree_path(
    cache_root,
    *,
    repo_owner: str = "o",
    repo_name: str = "r",
    base_sha: str = "b",
    config: str = "py",
):
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


def _stub_flow(monkeypatch, tmp_path, tasks: list[LCATask] | None = None):
    monkeypatch.setenv("HARNESS_PROJECT_DATASET", LANGFUSE_DATASET)
    monkeypatch.setenv("HARNESS_WORKTREE_CACHE_DIR", str(tmp_path / "worktrees"))
    monkeypatch.setattr(run_module.harness_run, "ensure_langfuse_auth", lambda: None)
    monkeypatch.setattr(run_module.harness_run, "flush_langfuse", lambda: None)
    monkeypatch.setattr(
        run_module,
        "fetch_localization_dataset",
        lambda *args, **kwargs: tasks or [_task()],
    )
    monkeypatch.setattr(
        run_module,
        "_compute_projection",
        lambda *args, **kwargs: {
            "total": {"planned": 0, "hard_cap": 0},
            "localization": {},
            "writer": {},
        },
    )
    monkeypatch.setattr(run_module, "_require_confirmation", lambda *args, **kwargs: None)


def test_typer_help_exposes_flow_surface_only():
    root = RUNNER.invoke(run_module.app, ["--help"])
    assert root.exit_code == 0
    assert "jc" in root.output
    assert "ic" in root.output
    assert "sync-hf" in root.output
    assert "baseline" not in root.output
    assert "experiment" not in root.output

    for args in (["jc", "run"], ["ic", "run"], ["ic", "optimize"], ["sync-hf"]):
        result = RUNNER.invoke(run_module.app, [*args, "--help"])
        assert result.exit_code == 0
        assert "--help" in result.output


@pytest.mark.parametrize("args", (["jc", "run"], ["ic", "run"], ["ic", "optimize"]))
def test_flow_help_keeps_dataset_flags_out_of_operator_surface(args):
    result = RUNNER.invoke(run_module.app, [*args, "--help"])

    assert result.exit_code == 0
    assert "--max-items" in result.output
    assert "--offset" in result.output
    assert "--session-id" in result.output
    assert "--fresh-worktrees" in result.output
    assert "--dataset" not in result.output
    assert "--config" not in result.output
    assert "--split" not in result.output
    assert "--revision" not in result.output


def test_flow_narratives_align_with_commands():
    assert set(FLOW_NARRATIVES) == {"jc run", "ic run", "ic optimize"}
    assert all(isinstance(narrative, FlowNarrative) for narrative in FLOW_NARRATIVES.values())
    optimize_steps = " ".join(IC_OPTIMIZE_NARRATIVE.steps)
    for expected in (
        "evaluate current IC behavior",
        "build writer feedback",
        "generate candidate policy",
        "pipeline correctness checks",
        "accept or reject",
    ):
        assert expected in optimize_steps


def test_task_summary_formatter_success_order(capsys):
    run_module.harness_run._print_task_summary(_task_summary("ic"))

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[TASK] ic identity=task-id"
    assert lines[1] == "status=success"
    assert lines[2] == "repo=/repo"
    assert lines[3] == "steps=1 tool_calls=3 successful_tool_calls=2 failed_tool_calls=1 generations=2"
    assert lines[4] == "tokens: fresh_input=10.00 cached_input=20.00 total_input=30.00 output=5.00 total=35.00"
    assert lines[5] == "result: predicted_files=4"
    assert lines[6] == "evaluation: gold_overlap_count=1.00"


def test_task_summary_formatter_failure_order(capsys):
    run_module.harness_run._print_task_summary(_task_summary("jc", status="failure"))

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[TASK] jc identity=task-id"
    assert lines[1] == "status=failure"
    assert lines[2] == "stage=backend_init"
    assert lines[3] == "repo=/repo"
    assert lines[4] == "steps=1 tool_calls=3 successful_tool_calls=2 failed_tool_calls=1 generations=2"
    assert lines[5] == "tokens: fresh_input=10.00 cached_input=20.00 total_input=30.00 output=5.00 total=35.00"
    assert lines[6] == "message=boom"


def test_run_summary_formatter_uses_task_aggregates(capsys):
    run_summary = RunSummary.from_task_summaries(
        backend="ic",
        selected_tasks_count=2,
        task_summaries=[_task_summary("ic")],
    )

    run_module.harness_run._print_run_summary(run_summary)

    out = capsys.readouterr().out
    assert "[RUN_SUMMARY] backend=ic selected=2 completed=1 successful=1 failed=0 success_rate=1.00" in out
    assert "tokens: fresh_input=10.00 cached_input=20.00 total_input=30.00 output=5.00 total=35.00 avg_total=35.00" in out
    assert "tool_calls=3 avg_tool_calls=3.00 generations=2 avg_generations=2.00" in out


def test_ic_optimize_help_narrates_evaluator_writer_pipeline_flow():
    result = RUNNER.invoke(run_module.app, ["ic", "optimize", "--help"])
    assert result.exit_code == 0
    assert "--max-workers" in result.output
    assert "isolated" in result.output
    assert "evaluate current IC behavior" in result.output
    assert "build writer feedback" in result.output
    assert "pipeline correctness checks" in result.output
    assert "accept" in result.output
    assert "reject" in result.output


def test_old_commands_are_not_registered():
    for command in ("baseline", "experiment"):
        result = RUNNER.invoke(run_module.app, [command, "--help"])
        assert result.exit_code != 0


def test_removed_flags_rejected():
    with pytest.raises(SystemExit):
        run_module.main(["ic", "run", "--mode", "localization"])
    with pytest.raises(SystemExit):
        run_module.main(["ic", "run", "--assume-input-tokens", "10"])
    with pytest.raises(SystemExit):
        run_module.main(["ic", "run", "--invalidate-baseline-cache"])


def test_sync_hf_uses_project_defaults_without_required_dataset_args(monkeypatch):
    calls: dict[str, object] = {}

    def fake_sync(**kwargs):
        calls.update(kwargs)
        return HFDatasetSyncResult(
            items=[],
            source_items=1,
            existing_items=0,
            created_items=1,
            skipped_items=0,
        )

    monkeypatch.setattr(run_module.harness_run, "ensure_langfuse_auth", lambda: None)
    monkeypatch.setattr(run_module.harness_run, "flush_langfuse", lambda: None)
    monkeypatch.setattr(
        run_module.harness_run,
        "sync_hf_localization_dataset_to_langfuse",
        fake_sync,
    )
    monkeypatch.setenv("HARNESS_PROJECT_DATASET", LANGFUSE_DATASET)

    result = RUNNER.invoke(run_module.app, ["sync-hf"])

    assert result.exit_code == 0
    assert calls["langfuse_dataset"] == LANGFUSE_DATASET
    assert calls["dataset_config"] == "py"
    assert calls["dataset_split"] == "dev"
    assert calls["revision"] is None


def test_sync_hf_optional_overrides_forward_ingestion_arguments(monkeypatch):
    calls: dict[str, object] = {}

    def fake_sync(**kwargs):
        calls.update(kwargs)
        return HFDatasetSyncResult(
            items=[],
            source_items=1,
            existing_items=0,
            created_items=1,
            skipped_items=0,
        )

    monkeypatch.setattr(run_module.harness_run, "ensure_langfuse_auth", lambda: None)
    monkeypatch.setattr(run_module.harness_run, "flush_langfuse", lambda: None)
    monkeypatch.setattr(
        run_module.harness_run,
        "sync_hf_localization_dataset_to_langfuse",
        fake_sync,
    )

    result = RUNNER.invoke(
        run_module.app,
        [
            "sync-hf",
            "--dataset",
            LANGFUSE_DATASET,
            "--config",
            "py",
            "--split",
            "dev",
            "--revision",
            "rev1",
        ],
    )
    assert result.exit_code == 0
    assert calls["langfuse_dataset"] == LANGFUSE_DATASET
    assert calls["dataset_config"] == "py"
    assert calls["dataset_split"] == "dev"
    assert calls["revision"] == "rev1"


@pytest.mark.parametrize("args", (["jc", "run"], ["ic", "run"]))
def test_run_flows_do_not_require_dataset_flags(args, monkeypatch, tmp_path):
    calls: dict[str, object] = {}
    _stub_flow(monkeypatch, tmp_path)

    def fake_evaluate(tasks, **kwargs):
        calls["tasks"] = tasks
        calls["runner"] = kwargs.get("runner")
        return LocalizationEvaluationResult()

    monkeypatch.setattr(run_module, "evaluate_localization_batch", fake_evaluate)

    run_module.main([*args, "--max-items", "1", "--yes"])

    assert calls["tasks"]
    assert callable(calls["runner"])


def test_flow_dataset_is_resolved_from_project_defaults(monkeypatch, tmp_path):
    fetch_calls: list[tuple[str, str | None]] = []
    _stub_flow(monkeypatch, tmp_path)
    monkeypatch.setenv("HARNESS_PROJECT_DATASET_VERSION", "v1")

    def fake_fetch(dataset_name, *, version=None):
        fetch_calls.append((dataset_name, version))
        return [_task()]

    monkeypatch.setattr(run_module, "fetch_localization_dataset", fake_fetch)
    monkeypatch.setattr(
        run_module,
        "evaluate_localization_batch",
        lambda *args, **kwargs: LocalizationEvaluationResult(),
    )

    run_module.main(["jc", "run", "--max-items", "1", "--yes"])

    assert fetch_calls == [(LANGFUSE_DATASET, "v1")]


def test_jc_and_ic_run_delegate_to_evaluation_backend(monkeypatch, tmp_path):
    calls: list[str] = []
    _stub_flow(monkeypatch, tmp_path)

    def fake_evaluate(tasks, **kwargs):
        runner = kwargs.get("runner")
        assert callable(runner)
        calls.append("called")
        return LocalizationEvaluationResult()

    monkeypatch.setattr(run_module, "evaluate_localization_batch", fake_evaluate)

    run_module.main(["jc", "run", "--max-items", "1", "--yes"])
    run_module.main(["ic", "run", "--max-items", "1", "--yes"])

    assert calls == ["called", "called"]


@pytest.mark.parametrize(
    ("args", "patch_name"),
    ((["jc", "run"], "run_jc_iteration"), (["ic", "run"], "run_ic_iteration")),
)
def test_run_backend_receives_materialized_repo_path(args, patch_name, monkeypatch, tmp_path):
    seen_repos: list[str | None] = []
    _stub_flow(monkeypatch, tmp_path)
    repo_path = tmp_path / "materialized"
    repo_path.mkdir()

    def fake_backend(task, **kwargs):
        seen_repos.append(task.repo)
        return SimpleNamespace(
            prediction=SimpleNamespace(predicted_files=["a.py"]),
            model_dump=lambda: {},
            backend=None,
            source=None,
        )

    monkeypatch.setattr(run_module, patch_name, fake_backend)

    def fake_evaluate(tasks, **kwargs):
        runner = kwargs["runner"]
        runner(tasks[0], str(repo_path), object())
        return LocalizationEvaluationResult()

    monkeypatch.setattr(run_module, "evaluate_localization_batch", fake_evaluate)

    run_module.main([*args, "--max-items", "1", "--yes"])

    assert seen_repos == [str(repo_path)]


def test_projection_only_skips_execution(monkeypatch, tmp_path):
    calls = {"evaluate": 0}
    _stub_flow(monkeypatch, tmp_path)

    def fake_evaluate(*args, **kwargs):
        calls["evaluate"] += 1
        return LocalizationEvaluationResult()

    monkeypatch.setattr(run_module, "evaluate_localization_batch", fake_evaluate)

    run_module.main(["ic", "run", "--projection-only", "--yes", "--max-items", "1"])

    assert calls["evaluate"] == 0


def test_fresh_worktrees_refreshes_selected_worktrees_only(monkeypatch, tmp_path):
    selected = _task("selected", "repo", "a" * 40)
    other = _task("other", "repo", "b" * 40)
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
    unrelated_file = tmp_path / "unrelated_cache" / "bundle.json"
    unrelated_file.parent.mkdir(parents=True)
    unrelated_file.write_text("{}")

    _stub_flow(monkeypatch, tmp_path, tasks=[selected, other])
    monkeypatch.setenv("HARNESS_WORKTREE_CACHE_DIR", str(checkout_cache_root))
    monkeypatch.setattr(
        run_module,
        "_select_tasks",
        lambda tasks, max_items, offset: (
            [selected],
            {
                "selected_count": 1,
                "offset": offset,
                "limit": max_items,
                "total_available": 2,
                "preview": [selected.task_id],
            },
        ),
    )
    monkeypatch.setattr(
        run_module,
        "evaluate_localization_batch",
        lambda *args, **kwargs: LocalizationEvaluationResult(),
    )

    run_module.main(["ic", "run", "--max-items", "1", "--yes", "--fresh-worktrees"])

    assert not selected_path.exists()
    assert other_path.exists()
    assert unrelated_file.exists()


def test_cli_fresh_session_per_invocation(monkeypatch, tmp_path, capsys):
    _stub_flow(monkeypatch, tmp_path)
    monkeypatch.setattr(
        run_module,
        "evaluate_localization_batch",
        lambda *args, **kwargs: LocalizationEvaluationResult(),
    )
    argv = ["ic", "run", "--max-items", "1", "--yes"]

    run_module.main(argv)
    first = capsys.readouterr().out
    run_module.main(argv)
    second = capsys.readouterr().out

    first_session = next(part for part in first.split() if part.startswith("session="))
    second_session = next(part for part in second.split() if part.startswith("session="))
    assert first_session != second_session


def test_cli_respects_explicit_session_id(monkeypatch, tmp_path, capsys):
    _stub_flow(monkeypatch, tmp_path)
    monkeypatch.setattr(
        run_module,
        "evaluate_localization_batch",
        lambda *args, **kwargs: LocalizationEvaluationResult(),
    )

    run_module.main(
        ["ic", "run", "--max-items", "1", "--yes", "--session-id", "explicit-session"]
    )

    assert "session=explicit-session" in capsys.readouterr().out


def test_ic_optimize_delegates_to_run_loop(monkeypatch, tmp_path):
    calls: list[LCATask] = []
    task = _task()
    _stub_flow(monkeypatch, tmp_path, tasks=[task])
    monkeypatch.setattr(
        run_module.harness_run,
        "_materialize_tasks_for_optimization",
        lambda tasks, worktree_manager: [task.with_repo(str(tmp_path))],
    )
    monkeypatch.setattr(
        run_module.harness_run,
        "_prepare_isolated_optimization_workspace",
        lambda task_arg, workspace_base, ordinal: (
            tmp_path / f"workspace-{ordinal}",
            task_arg.with_repo(str(tmp_path / f"repo-{ordinal}")),
        ),
    )

    def fake_run_loop(task_arg, **kwargs):
        calls.append(task_arg)
        assert kwargs["session_id"]
        return [SimpleNamespace(iteration=0)]

    monkeypatch.setattr(run_module, "run_loop", fake_run_loop)

    run_module.main(["ic", "optimize", "--max-items", "1", "--yes"])

    assert len(calls) == 1
    assert calls[0].repo == str(tmp_path / "repo-1")


def test_ic_optimize_honors_max_workers_with_isolated_workspaces(monkeypatch, tmp_path, capsys):
    tasks = [_task(repo_name="r1"), _task(repo_name="r2")]
    _stub_flow(monkeypatch, tmp_path, tasks=tasks)
    materialized = [
        task.with_repo(str(tmp_path / f"source-{idx}"))
        for idx, task in enumerate(tasks, start=1)
    ]
    monkeypatch.setattr(
        run_module.harness_run,
        "_materialize_tasks_for_optimization",
        lambda tasks_arg, worktree_manager: materialized,
    )
    monkeypatch.setattr(
        run_module.harness_run,
        "_prepare_isolated_optimization_workspace",
        lambda task_arg, workspace_base, ordinal: (
            tmp_path / f"workspace-{ordinal}",
            task_arg.with_repo(str(tmp_path / f"isolated-repo-{ordinal}")),
        ),
    )

    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_run_loop(task_arg, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return [SimpleNamespace(iteration=0)]

    monkeypatch.setattr(run_module, "run_loop", fake_run_loop)

    run_module.main(["ic", "optimize", "--max-items", "2", "--max-workers", "2", "--yes"])

    out = capsys.readouterr().out
    assert "execution=parallel effective_workers=2 isolation=per-task-workspace" in out
    assert "started=2 completed=2 failed=0 remaining=0 total=2" in out
    assert max_active == 2


def test_ic_optimize_traces_parent_and_task_spans(monkeypatch, tmp_path):
    task = _task()
    starts: list[dict[str, object]] = []
    run_loop_parents: list[object | None] = []
    _stub_flow(monkeypatch, tmp_path, tasks=[task])
    monkeypatch.setattr(
        run_module.harness_run,
        "_materialize_tasks_for_optimization",
        lambda tasks, worktree_manager: [task.with_repo(str(tmp_path))],
    )
    monkeypatch.setattr(
        run_module.harness_run,
        "_prepare_isolated_optimization_workspace",
        lambda task_arg, workspace_base, ordinal: (
            tmp_path / f"workspace-{ordinal}",
            task_arg.with_repo(str(tmp_path / f"repo-{ordinal}")),
        ),
    )

    @contextmanager
    def span_cm(*args, **kwargs):
        name = kwargs.get("name") or (args[0] if args else None)
        span = type(
            "S",
            (),
            {
                "id": f"span-{name}",
                "name": name,
                "update": lambda self, **kw: None,
                "end": lambda self, **kw: None,
            },
        )()
        starts.append(
            {
                "name": name,
                "parent": kwargs.get("parent"),
                "metadata": kwargs.get("metadata"),
                "span": span,
            }
        )
        yield span

    monkeypatch.setattr(run_module.harness_run, "start_observation", span_cm)

    def fake_run_loop(*args, **kwargs):
        run_loop_parents.append(kwargs.get("parent_trace"))
        return [SimpleNamespace(iteration=0)]

    monkeypatch.setattr(run_module, "run_loop", fake_run_loop)

    run_module.main(["ic", "optimize", "--max-items", "1", "--yes"])

    coordinator = next(start for start in starts if start["name"] == "ic_optimize_dataset")
    task_start = next(start for start in starts if start["name"] == "optimize_task")
    assert coordinator["parent"] is None
    assert task_start["parent"] is None
    assert run_loop_parents == [task_start["span"]]
    assert task_start["span"] is not coordinator["span"]


def test_select_tasks_is_deterministic_and_offsets():
    tasks = [
        _task("o2", "r2", "b2"),
        _task("o1", "r1", "b1"),
    ]
    selected, info = run_module._select_tasks(tasks, max_items=1, offset=0)
    assert [t.identity.task_id() for t in selected] == [
        _task("o1", "r1", "b1").identity.task_id()
    ]
    assert info["selected_count"] == 1
    selected_offset, info_offset = run_module._select_tasks(
        tasks, max_items=1, offset=1
    )
    assert [t.identity.task_id() for t in selected_offset] == [
        _task("o2", "r2", "b2").identity.task_id()
    ]
    assert info_offset["offset"] == 1


def test_projection_fail_closed_on_unknown_pricing_without_override(monkeypatch):
    monkeypatch.setattr(run_module, "cost_details_for_usage", lambda *args, **kwargs: None)
    with pytest.raises(ValueError):
        run_module._compute_projection(1)


def test_projection_with_override(monkeypatch):
    def fake_cost(model, usage):
        return {"total": (usage.get("input", 0) + usage.get("output", 0)) * 0.000001}

    monkeypatch.setattr(run_module, "cost_details_for_usage", fake_cost)
    proj = run_module._compute_projection(3)
    assert proj["total"]["planned"] < proj["total"]["hard_cap"]
    assert proj["localization"]["planned"]["cost"] < proj["localization"]["hard_cap"]["cost"]


def test_confirmation_fallback_line_prompt(monkeypatch):
    monkeypatch.setattr(run_module, "termios", None)
    monkeypatch.setattr(run_module, "tty", None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "space")
    run_module._require_confirmation(skip_confirmation=False, selection={}, projection={})


def test_confirmation_non_interactive_rejected(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(RuntimeError):
        run_module._require_confirmation(skip_confirmation=False, selection={}, projection={})
