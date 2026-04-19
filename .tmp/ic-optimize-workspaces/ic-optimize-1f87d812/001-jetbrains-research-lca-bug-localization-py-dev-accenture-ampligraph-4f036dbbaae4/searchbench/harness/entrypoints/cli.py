"""
Localization-first CLI entrypoint: declare project flows, select fixed tasks, and run LCA bug localization.
Adds flow guardrails: deterministic selection, cost projection, confirmation gating, and automation safety.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Annotated, Any, Callable, Iterable, Iterator, Sequence

try:
    import termios  # type: ignore
    import tty  # type: ignore
except ImportError:
    tty = None  # type: ignore
    termios = None  # type: ignore

import typer
from pydantic import BaseModel, ConfigDict

from harness.agents.localizer import run_ic_iteration, run_jc_iteration
from harness.entrypoints.flows import (
    IC_OPTIMIZE_NARRATIVE,
    IC_RUN_NARRATIVE,
    JC_RUN_NARRATIVE,
    flow_help,
)
from harness.localization.materialization.worktree import WorktreeManager
from harness.localization.models import LCATask
from harness.localization.runtime.evaluate import (
    LocalizationEvaluationResult,
    LocalizationRunner,
)
from harness.localization.runtime.evaluate import (
    evaluate_localization_batch as _evaluate_localization_batch,
)
from harness.orchestration.machine import run_loop
from harness.orchestration.runtime_context import isolated_optimization_workspace
from harness.orchestration import writer as policy_writer
from harness.telemetry.hosted.datasets import fetch_localization_dataset
from harness.telemetry.hosted.hf_lca import (
    HFDatasetLoadError,
    sync_hf_localization_dataset_to_langfuse,
)
from harness.telemetry.observability_models import RunSummary, TaskRunSummary
from harness.telemetry.tracing import (
    ensure_langfuse_auth,
    flush_langfuse,
    propagate_context,
    start_observation,
)
from harness.telemetry.tracing.cerebras_pricing import cost_details_for_usage
from harness.telemetry.tracing.session_policy import SessionConfig

WARNING_THRESHOLD = 50
HF_DATASET_NAME = "JetBrains-Research/lca-bug-localization"
PROJECT_DATASET_NAME = "lca-localization"
PROJECT_HF_DATASET_CONFIG = "py"
PROJECT_HF_DATASET_SPLIT = "dev"
LANGFUSE_DATASET_STORE = "langfuse"
REMOVED_FLAGS = {
    "--mode",
    "--dataset-source",
    "--dataset_source",
    "--repo-owner",
    "--repo-name",
    "--base-sha",
    "--repo-path",
    "--issue-url",
    "--pull-url",
    "--issue-title",
    "--issue-body",
    "--diff",
    "--diff-url",
    "--changed-file",
    "--baseline-path",
    "--assume-input-tokens",
    "--assume-output-tokens",
    "--assume-calls-per-item",
    "--assume-model",
    "--assume-price-input-per-million",
    "--assume-price-output-per-million",
}
REMOVED_FRESHNESS_FLAGS = {"--invalidate-baseline-cache"}

# Runner shape and iteration bounds
DEFAULT_OPTIMIZATION_ITERATIONS = 5
MAX_POLICY_REPAIRS_PER_ITERATION = 3
WRITER_INTERNAL_ATTEMPTS_PER_REPAIR = 3

# Localization hard caps (llama3.1-8b)
LOCALIZATION_MODEL = "gpt-oss-120b"
LOCALIZATION_PLANNED_CALLS_PER_ITEM = 6  # 5 explore + 1 finalize
LOCALIZATION_MAX_CALLS_PER_ITEM = 11  # retry-aware
LOCALIZATION_INPUT_TOKENS_PER_ITEM_PLANNED = 75_000
LOCALIZATION_OUTPUT_TOKENS_PER_ITEM_PLANNED = 10_000
LOCALIZATION_INPUT_TOKENS_PER_ITEM_MAX = 110_000
LOCALIZATION_OUTPUT_TOKENS_PER_ITEM_MAX = 14_000

# Writer hard caps (gpt-oss-120b)
WRITER_MODEL = "gpt-oss-120b"
WRITER_INPUT_TOKENS_PER_CALL = 20_000
WRITER_OUTPUT_TOKENS_PER_CALL = 4_096
WRITER_MAX_CALLS_PER_ITERATION = (
    MAX_POLICY_REPAIRS_PER_ITERATION * WRITER_INTERNAL_ATTEMPTS_PER_REPAIR
)  # 9
WRITER_MAX_CALLS_PER_RUN = (
    DEFAULT_OPTIMIZATION_ITERATIONS * WRITER_MAX_CALLS_PER_ITERATION
)  # 45
WRITER_PLANNED_CALLS_PER_ITERATION = 1
WRITER_PLANNED_CALLS_PER_RUN = (
    DEFAULT_OPTIMIZATION_ITERATIONS * WRITER_PLANNED_CALLS_PER_ITERATION
)
WRITER_MAX_INPUT_TOKENS_PER_ITERATION = (
    WRITER_MAX_CALLS_PER_ITERATION * WRITER_INPUT_TOKENS_PER_CALL
)
WRITER_MAX_OUTPUT_TOKENS_PER_ITERATION = (
    WRITER_MAX_CALLS_PER_ITERATION * WRITER_OUTPUT_TOKENS_PER_CALL
)
WRITER_MAX_INPUT_TOKENS_PER_RUN = (
    DEFAULT_OPTIMIZATION_ITERATIONS * WRITER_MAX_INPUT_TOKENS_PER_ITERATION
)
WRITER_MAX_OUTPUT_TOKENS_PER_RUN = (
    DEFAULT_OPTIMIZATION_ITERATIONS * WRITER_MAX_OUTPUT_TOKENS_PER_ITERATION
)

app = typer.Typer(
    help="Project flow runner for JC, IC, and IC optimization",
    no_args_is_help=True,
)
jc_app = typer.Typer(help="JCodeMunch flows", no_args_is_help=True)
ic_app = typer.Typer(help="Iterative Context flows", no_args_is_help=True)
app.add_typer(jc_app, name="jc")
app.add_typer(ic_app, name="ic")

MaxItemsOption = Annotated[
    int | None,
    typer.Option("--max-items", help="Maximum number of project tasks to evaluate"),
]
OffsetOption = Annotated[
    int,
    typer.Option("--offset", help="Offset into the fixed task set before selection"),
]
MaxWorkersOption = Annotated[
    int,
    typer.Option(
        "--max-workers",
        help="Maximum parallel workers for selected project task execution (>=1)",
    ),
]
OptimizeMaxWorkersOption = Annotated[
    int,
    typer.Option(
        "--max-workers",
        help="Maximum parallel per-task optimization workers; each worker uses an isolated workspace and policy file (>=1)",
    ),
]
ProjectionOnlyOption = Annotated[
    bool,
    typer.Option(
        "--projection-only",
        help="Compute selection and cost projection, then exit after confirmation without executing runs",
    ),
]
YesOption = Annotated[
    bool,
    typer.Option(
        "--yes", help="Bypass interactive confirmation (requires --max-items)"
    ),
]
SessionIdOption = Annotated[
    str | None,
    typer.Option("--session-id", help="Optional explicit session id"),
]
FreshWorktreesOption = Annotated[
    bool,
    typer.Option(
        "--fresh-worktrees",
        help="Refresh selected repository worktrees before execution",
    ),
]


class ProjectDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str | None = None
    hf_config: str = PROJECT_HF_DATASET_CONFIG
    hf_split: str = PROJECT_HF_DATASET_SPLIT
    hf_revision: str | None = None


@contextmanager
def _raw_mode(file_obj: IO[Any]) -> Iterator[None]:
    """
    Best-effort raw mode for single-key reads; falls back to normal mode on failure.
    """
    if termios is None or tty is None:
        yield
        return
    fd = file_obj.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def evaluate_localization_batch(
    tasks: Sequence[LCATask],
    *,
    dataset_provenance: str | None = None,
    worktree_manager: WorktreeManager | None = None,
    parent_trace: object | None = None,
    runner: LocalizationRunner | None = None,
):
    """
    Thin wrapper to keep monkeypatch targets stable for tests while deferring import cycles.
    """
    return _evaluate_localization_batch(
        tasks,
        dataset_provenance=dataset_provenance,
        worktree_manager=worktree_manager,
        parent_trace=parent_trace,
        runner=runner,
    )


def _reject_removed_flags(argv: Sequence[str]) -> None:
    if any(flag in argv for flag in REMOVED_FRESHNESS_FLAGS):
        raise SystemExit(
            "This freshness flag is no longer supported; use --fresh-worktrees."
        )
    for flag in REMOVED_FLAGS:
        if flag in argv:
            raise SystemExit(
                f"{flag} is no longer supported; use jc run, ic run, ic optimize, or sync-hf."
            )


def _print_mat_summary(prefix: str, identity: Any, materialization: Any) -> None:
    if materialization:
        print(f"[MAT] {prefix} {identity}: {materialization.summary()}")


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _sort_tasks_deterministically(tasks: Iterable[LCATask]) -> list[LCATask]:
    return sorted(
        list(tasks), key=lambda t: getattr(t, "task_id", None) or t.identity.task_id()
    )


def _build_flow_session_config(
    command: str, explicit_session_id: str | None
) -> SessionConfig:
    """
    Build a fresh run-scope SessionConfig for this CLI invocation.
    """
    session_prefix = command.replace(" ", "-")
    session_id = explicit_session_id or f"{session_prefix}-{uuid.uuid4().hex[:8]}"
    return SessionConfig(
        session_id=session_id,
        session_scope="run",
        allow_generated_fallback=False,
    )


def _refresh_worktree_checkouts_for_tasks(
    tasks: Sequence[LCATask], worktree_manager: WorktreeManager
) -> list[str]:
    """
    Best-effort targeted refresh of repo worktree caches for selected tasks.
    """
    events = [str(event) for event in worktree_manager.invalidate_tasks(tasks)]
    print(f"[CACHE] Repo worktree cache refreshed for {len(tasks)} selected task(s)")
    for event in events:
        print(f"[CACHE] {event}")
    return events


def resolve_project_dataset() -> ProjectDataset:
    version = os.environ.get("HARNESS_PROJECT_DATASET_VERSION") or None
    return ProjectDataset(
        name=os.environ.get("HARNESS_PROJECT_DATASET", PROJECT_DATASET_NAME),
        version=version,
        hf_config=os.environ.get("HARNESS_PROJECT_HF_CONFIG", PROJECT_HF_DATASET_CONFIG),
        hf_split=os.environ.get("HARNESS_PROJECT_HF_SPLIT", PROJECT_HF_DATASET_SPLIT),
        hf_revision=os.environ.get("HARNESS_PROJECT_HF_REVISION") or None,
    )


def _select_tasks(
    tasks: list[LCATask], max_items: int | None, offset: int
) -> tuple[list[LCATask], dict[str, object]]:
    if offset < 0:
        raise ValueError("--offset must be >= 0")
    sorted_tasks = _sort_tasks_deterministically(tasks)
    total_available = len(sorted_tasks)
    if max_items is not None and max_items <= 0:
        raise ValueError("--max-items must be > 0")
    start = min(offset, total_available)
    end = (
        total_available
        if max_items is None
        else min(start + max_items, total_available)
    )
    selected = sorted_tasks[start:end]
    preview = [
        getattr(t, "task_id", None) or t.identity.task_id() for t in selected[:5]
    ]
    selection_info: dict[str, object] = {
        "total_available": total_available,
        "offset": offset,
        "limit": max_items,
        "selected_count": len(selected),
        "preview": preview,
    }
    return selected, selection_info


def _print_selection(
    dataset: str,
    version: str | None,
    store: str,
    selection: dict[str, object],
) -> None:
    limit = selection.get("limit")
    offset = selection.get("offset")
    selected_count = selection.get("selected_count")
    total_available = selection.get("total_available")
    preview = selection.get("preview") or []
    print(
        "[SELECT] dataset=%s version=%s store=%s offset=%s limit=%s selected=%s total=%s"
        % (
            dataset,
            version,
            store,
            offset,
            limit,
            selected_count,
            total_available,
        )
    )
    if preview:
        print(f"[SELECT] preview identities (up to 5): {preview}")


def _warn_if_unbounded(selection: dict[str, object], max_items: int | None) -> None:
    selected_count = _coerce_int(selection.get("selected_count"))
    total_available = _coerce_int(selection.get("total_available"))
    if max_items is None and total_available > WARNING_THRESHOLD:
        print(
            f"[WARN] No --max-items provided; {total_available} items available. Run is unbounded."
        )
    if selected_count > WARNING_THRESHOLD:
        print(f"[WARN] Large selection: {selected_count} items will run.")


def _price_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = cost_details_for_usage(
        model, {"input": input_tokens, "output": output_tokens}
    )
    if not costs or "total" not in costs:
        raise ValueError(
            f"Unknown pricing for model/provider '{model}'; ensure Cerebras pricing covers it."
        )
    return float(costs.get("total") or 0.0)


def _compute_projection(selected_count: int) -> dict[str, Any]:
    if selected_count <= 0:
        raise ValueError(
            "No items selected for projection; adjust --max-items/--offset."
        )

    loc_input_planned = selected_count * LOCALIZATION_INPUT_TOKENS_PER_ITEM_PLANNED
    loc_output_planned = selected_count * LOCALIZATION_OUTPUT_TOKENS_PER_ITEM_PLANNED
    loc_input_max = selected_count * LOCALIZATION_INPUT_TOKENS_PER_ITEM_MAX
    loc_output_max = selected_count * LOCALIZATION_OUTPUT_TOKENS_PER_ITEM_MAX

    writer_input_planned = WRITER_PLANNED_CALLS_PER_RUN * WRITER_INPUT_TOKENS_PER_CALL
    writer_output_planned = WRITER_PLANNED_CALLS_PER_RUN * WRITER_OUTPUT_TOKENS_PER_CALL
    writer_input_max = WRITER_MAX_CALLS_PER_RUN * WRITER_INPUT_TOKENS_PER_CALL
    writer_output_max = WRITER_MAX_CALLS_PER_RUN * WRITER_OUTPUT_TOKENS_PER_CALL

    loc_cost_planned = _price_tokens(
        LOCALIZATION_MODEL, loc_input_planned, loc_output_planned
    )
    loc_cost_max = _price_tokens(LOCALIZATION_MODEL, loc_input_max, loc_output_max)
    writer_cost_planned = _price_tokens(
        WRITER_MODEL, writer_input_planned, writer_output_planned
    )
    writer_cost_max = _price_tokens(WRITER_MODEL, writer_input_max, writer_output_max)

    return {
        "localization": {
            "model": LOCALIZATION_MODEL,
            "planned": {
                "input_tokens": loc_input_planned,
                "output_tokens": loc_output_planned,
                "cost": loc_cost_planned,
            },
            "hard_cap": {
                "input_tokens": loc_input_max,
                "output_tokens": loc_output_max,
                "cost": loc_cost_max,
            },
        },
        "writer": {
            "model": WRITER_MODEL,
            "planned": {
                "input_tokens": writer_input_planned,
                "output_tokens": writer_output_planned,
                "cost": writer_cost_planned,
            },
            "hard_cap": {
                "input_tokens": writer_input_max,
                "output_tokens": writer_output_max,
                "cost": writer_cost_max,
            },
        },
        "total": {
            "planned": loc_cost_planned + writer_cost_planned,
            "hard_cap": loc_cost_max + writer_cost_max,
        },
    }


def _print_projection(selection: dict[str, object], projection: dict[str, Any]) -> None:
    print(f"[PROJECTION] selected_items={selection.get('selected_count')}")
    loc = projection.get("localization") or {}
    writer = projection.get("writer") or {}
    total = projection.get("total") or {}

    def _fmt_block(block: dict[str, Any]) -> tuple[Any, ...]:
        planned = block.get("planned") or {}
        hard_cap = block.get("hard_cap") or {}
        return (
            block.get("model", "unknown"),
            planned.get("input_tokens", 0),
            planned.get("output_tokens", 0),
            float(planned.get("cost", 0) or 0.0),
            hard_cap.get("input_tokens", 0),
            hard_cap.get("output_tokens", 0),
            float(hard_cap.get("cost", 0) or 0.0),
        )

    loc_values = _fmt_block(loc)
    writer_values = _fmt_block(writer)

    print(
        "[PROJECTION] localization model=%s planned_input=%s planned_output=%s planned_cost=%.4f hardcap_input=%s hardcap_output=%s hardcap_cost=%.4f"
        % loc_values
    )
    print(
        "[PROJECTION] writer model=%s planned_input=%s planned_output=%s planned_cost=%.4f hardcap_input=%s hardcap_output=%s hardcap_cost=%.4f"
        % writer_values
    )
    print(
        "[PROJECTION] totals planned=%.4f hardcap=%.4f"
        % (
            float(total.get("planned", 0) or 0.0),
            float(total.get("hard_cap", 0) or 0.0),
        )
    )


def _require_confirmation(
    skip_confirmation: bool,
    selection: dict[str, object],
    projection: dict[str, float | int | str],
) -> None:
    if skip_confirmation:
        print("[CONFIRM] --yes set; proceeding without interactive confirmation.")
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Interactive confirmation required on TTY; rerun with --yes and --max-items if automation is intended."
        )
    prompt = "Press SPACE to continue with the run (or Ctrl+C to cancel). "
    try:
        with _raw_mode(sys.stdin):
            sys.stdout.write(prompt)
            sys.stdout.flush()
            ch = sys.stdin.read(1)
            if ch == " ":
                print("\n[CONFIRM] Proceeding.")
                return
            else:
                print("\n[CONFIRM] Input not accepted; please press SPACE to continue.")
                return _require_confirmation(skip_confirmation, selection, projection)
    except Exception:
        # Fallback to line-based confirmation
        response = (
            input("Type 'space' to confirm and continue (or Ctrl+C to cancel): ")
            .strip()
            .lower()
        )
        if response == "space" or response == "yes":
            print("[CONFIRM] Proceeding.")
            return
        print("[CONFIRM] Input not accepted; please confirm explicitly.")
        return _require_confirmation(skip_confirmation, selection, projection)


def _prepare_flow_run(
    *,
    command: str,
    max_items: int | None,
    offset: int,
    max_workers: int,
    yes: bool,
    projection_only: bool,
    session_id: str | None,
    fresh_worktrees: bool,
) -> tuple[
    ProjectDataset,
    WorktreeManager,
    SessionConfig,
    list[LCATask],
    dict[str, object],
    dict[str, Any],
    bool,
]:
    # Fail fast on Langfuse auth/base-url issues in the operator flow path.
    ensure_langfuse_auth()
    if max_workers <= 0:
        raise ValueError("--max-workers must be a positive integer")
    project_dataset = resolve_project_dataset()
    worktree_manager = WorktreeManager()
    session_cfg = _build_flow_session_config(
        command,
        session_id,
    )
    tasks = fetch_localization_dataset(
        project_dataset.name,
        version=project_dataset.version,
    )
    if not tasks:
        raise RuntimeError(f"No localization tasks available for {command} run")
    selected_tasks, selection_info = _select_tasks(tasks, max_items, offset)
    _warn_if_unbounded(selection_info, max_items)
    _print_selection(
        project_dataset.name,
        project_dataset.version,
        LANGFUSE_DATASET_STORE,
        selection_info,
    )
    projection = _compute_projection(_coerce_int(selection_info.get("selected_count")))
    _print_projection(selection_info, projection)
    if yes and not max_items:
        raise ValueError("--yes requires --max-items to be set for safety.")
    _require_confirmation(yes, selection_info, projection)
    if projection_only:
        print("[RUN] Projection-only mode: exiting before execution.")
    elif fresh_worktrees:
        _refresh_worktree_checkouts_for_tasks(selected_tasks, worktree_manager)
    return (
        project_dataset,
        worktree_manager,
        session_cfg,
        selected_tasks,
        selection_info,
        projection,
        projection_only,
    )


def _run_with_langfuse_finally(action: Callable[[], Any]) -> Any:
    try:
        return action()
    except HFDatasetLoadError as exc:
        print(f"[ERROR] HF dataset load failed (category={exc.category}): {exc}")
        raise
    except RuntimeError as exc:
        # Surface materialization failures or other runtime issues
        print(f"[ERROR] {exc}")
        raise
    finally:
        flush_langfuse()


def _runner_for_backend(backend: str) -> LocalizationRunner:
    if backend == "jc":
        return lambda task, repo_path, parent_trace: run_jc_iteration(
            task.with_repo(repo_path), parent_trace=parent_trace
        )
    if backend == "ic":
        return lambda task, repo_path, parent_trace: run_ic_iteration(
            task.with_repo(repo_path), parent_trace=parent_trace
        )
    raise ValueError(f"Unknown backend '{backend}'")


def _print_evaluation_summary(
    prefix: str, result: LocalizationEvaluationResult
) -> None:
    print(f"[RUN] {prefix} localization items completed: {len(result.items)}")
    if result.failure is not None:
        category = getattr(result.failure.category, "value", result.failure.category)
        print(
            f"[RUN] {prefix} run reported failure category={category} task_id={result.failure.task_id} message={result.failure.message}"
        )
        if result.failure.observability_summary is not None:
            _print_task_summary(result.failure.observability_summary)
    for item in result.items:
        if item.observability_summary is not None:
            _print_task_summary(item.observability_summary)
        _print_mat_summary(prefix, item.task.task_id, item.materialization)
    if result.run_summary is not None:
        _print_run_summary(result.run_summary)


def _fmt_value(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_task_summary(summary: TaskRunSummary) -> None:
    print(f"[TASK] {summary.backend} identity={summary.identity}")
    print(f"status={summary.status}")
    if summary.status == "failure" or summary.stage:
        print(f"stage={summary.stage or 'unknown'}")
    print(f"repo={summary.repo or 'unknown'}")
    print(
        "steps=%s tool_calls=%s successful_tool_calls=%s failed_tool_calls=%s generations=%s"
        % (
            _fmt_value(summary.step_count),
            _fmt_value(summary.tool_calls_count),
            _fmt_value(summary.successful_tool_calls_count),
            _fmt_value(summary.failed_tool_calls_count),
            _fmt_value(summary.generation_count),
        )
    )
    print(
        "tokens: fresh_input=%s cached_input=%s total_input=%s output=%s total=%s"
        % (
            _fmt_value(summary.tokens.fresh_input_tokens),
            _fmt_value(summary.tokens.cached_input_tokens),
            _fmt_value(summary.tokens.total_input_tokens),
            _fmt_value(summary.tokens.output_tokens),
            _fmt_value(summary.tokens.total_tokens),
        )
    )
    if summary.status == "success":
        print(f"result: predicted_files={summary.predicted_files_count}")
        if summary.evaluation:
            metrics = " ".join(
                f"{key}={_fmt_value(value)}" for key, value in sorted(summary.evaluation.items())
            )
            print(f"evaluation: {metrics}")
    if summary.status == "failure":
        print(f"message={summary.message or 'unknown'}")


def _print_run_summary(summary: RunSummary) -> None:
    print(
        "[RUN_SUMMARY] backend=%s selected=%s completed=%s successful=%s failed=%s success_rate=%s"
        % (
            summary.backend,
            summary.selected_tasks_count,
            summary.completed_tasks_count,
            summary.successful_tasks_count,
            summary.failed_tasks_count,
            _fmt_value(summary.success_rate),
        )
    )
    print(
        "[RUN_SUMMARY] tokens: fresh_input=%s cached_input=%s total_input=%s output=%s total=%s avg_total=%s"
        % (
            _fmt_value(summary.tokens.fresh_input_tokens),
            _fmt_value(summary.tokens.cached_input_tokens),
            _fmt_value(summary.tokens.total_input_tokens),
            _fmt_value(summary.tokens.output_tokens),
            _fmt_value(summary.tokens.total_tokens),
            _fmt_value(summary.average_tokens_per_task.total_tokens),
        )
    )
    print(
        "[RUN_SUMMARY] tool_calls=%s avg_tool_calls=%s generations=%s avg_generations=%s duration=%s avg_task_duration=%s"
        % (
            summary.total_tool_calls_count,
            _fmt_value(summary.average_tool_calls_per_task),
            summary.total_generations_count,
            _fmt_value(summary.average_generations_per_task),
            _fmt_value(summary.total_duration_seconds),
            _fmt_value(summary.average_task_duration_seconds),
        )
    )


def _materialize_tasks_for_optimization(
    tasks: Sequence[LCATask], worktree_manager: WorktreeManager
) -> list[LCATask]:
    materialized: list[LCATask] = []
    for task in tasks:
        request = worktree_manager._request_for_task(task)  # pyright: ignore[reportPrivateUsage]
        result = worktree_manager.get_or_create(request)
        if result.local_path is None:
            raise RuntimeError(
                f"Worktree materialization did not produce a path for {task.task_id}"
            )
        materialized.append(task.with_repo(str(result.local_path)))
    return materialized


@dataclass(frozen=True)
class OptimizeTaskResult:
    task: LCATask
    ordinal: int
    total: int
    status: str
    history: list[Any]
    duration_seconds: float
    workspace_root: Path | None = None
    isolated_repo: str | None = None
    error: str | None = None


def _optimize_workspace_base(session_id: str) -> Path:
    return Path(".tmp") / "ic-optimize-workspaces" / session_id


def _safe_task_slug(task: LCATask) -> str:
    raw = task.task_id
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in raw.lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80] or "task"


def _copy_tree_for_optimize(source: Path, destination: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        ".venv",
        ".tmp",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "__pycache__",
        "repomix-output.xml",
    )
    shutil.copytree(source, destination, symlinks=True, ignore=ignore)


def _prepare_isolated_optimization_workspace(
    task: LCATask,
    *,
    workspace_base: Path,
    ordinal: int,
) -> tuple[Path, LCATask]:
    source_root = Path(__file__).resolve().parents[2]
    task_root = workspace_base / f"{ordinal:03d}-{_safe_task_slug(task)}"
    harness_workspace = task_root / "searchbench"
    _copy_tree_for_optimize(source_root, harness_workspace)

    isolated_task = task
    if task.repo:
        source_repo = Path(task.repo)
        if source_repo.exists():
            isolated_repo = task_root / "target-repo"
            _copy_tree_for_optimize(source_repo, isolated_repo)
            isolated_task = task.with_repo(str(isolated_repo))
    return harness_workspace, isolated_task


def _accepted_policy_candidates(
    results: Sequence[OptimizeTaskResult],
) -> list[tuple[float, OptimizeTaskResult, Any]]:
    candidates: list[tuple[float, OptimizeTaskResult, Any]] = []
    for result in results:
        for record in result.history:
            accepted = getattr(record, "accepted_policy", None)
            if accepted is None:
                continue
            metrics = getattr(record, "metrics", None)
            metrics_get = getattr(metrics, "get", None)
            score_obj = metrics_get("score") if callable(metrics_get) else None
            score = float(score_obj) if isinstance(score_obj, (int, float)) else 0.0
            candidates.append((score, result, accepted))
    return candidates


def _apply_best_accepted_policy(
    results: Sequence[OptimizeTaskResult],
    parent_span: object | None,
) -> tuple[bool, str | None]:
    candidates = _accepted_policy_candidates(results)
    if not candidates:
        with start_observation(
            name="policy_acceptance",
            parent=parent_span,
            metadata={"status": "no_accepted_policy"},
        ):
            pass
        return False, None
    score, result, accepted = max(candidates, key=lambda item: item[0])
    policy_writer._write_policy(accepted.code)  # pyright: ignore[reportPrivateUsage]
    metadata = {
        "status": "accepted",
        "source_task": result.task.task_id,
        "source_ordinal": result.ordinal,
        "score": score,
        "workspace_root": str(result.workspace_root) if result.workspace_root else None,
    }
    with start_observation(
        name="policy_acceptance",
        parent=parent_span,
        metadata=metadata,
    ):
        pass
    return True, result.task.task_id


def _print_optimize_progress(
    *,
    started: int,
    completed: int,
    failed: int,
    total: int,
) -> None:
    remaining = max(0, total - completed - failed)
    print(
        "[RUN] ic optimize progress started=%s completed=%s failed=%s remaining=%s total=%s"
        % (started, completed, failed, remaining, total)
    )


def _run_localization_flow_command(
    *,
    command_name: str,
    backend: str,
    max_items: int | None,
    offset: int,
    max_workers: int,
    projection_only: bool,
    yes: bool,
    session_id: str | None,
    fresh_worktrees: bool,
) -> None:
    def _action() -> None:
        print(f"[RUN] {command_name}")
        (
            project_dataset,
            worktree_manager,
            session_cfg,
            selected_tasks,
            selection_info,
            projection,
            should_exit,
        ) = _prepare_flow_run(
            command=command_name,
            max_items=max_items,
            offset=offset,
            max_workers=max_workers,
            yes=yes,
            projection_only=projection_only,
            session_id=session_id,
            fresh_worktrees=fresh_worktrees,
        )
        if should_exit:
            return
        with propagate_context(
            trace_name=command_name.replace(" ", "_"),
            session_id=session_cfg.session_id,
            metadata={
                "command": command_name,
                "dataset": project_dataset.name,
                "dataset_version": project_dataset.version,
                "selection": selection_info,
                "projection": projection,
            },
        ):
            with start_observation(
                name=f"{backend}_run_dataset",
                metadata={
                    "command": command_name,
                    "dataset": project_dataset.name,
                    "dataset_version": project_dataset.version,
                    "selected_count": selection_info.get("selected_count"),
                },
            ) as dataset_span:
                result = evaluate_localization_batch(
                    selected_tasks,
                    dataset_provenance=LANGFUSE_DATASET_STORE,
                    worktree_manager=worktree_manager,
                    parent_trace=dataset_span,
                    runner=_runner_for_backend(backend),
                )
                if result.run_summary is not None:
                    try:
                        dataset_span.update(
                            metadata={
                                "run_summary": result.run_summary.model_dump(
                                    mode="json"
                                )
                            }
                        )
                    except Exception:
                        pass
        print(f"[RUN] session={session_cfg.session_id} dataset={project_dataset.name}")
        _print_evaluation_summary(backend, result)

    _run_with_langfuse_finally(_action)


def _run_ic_optimize_command(
    *,
    max_items: int | None,
    offset: int,
    max_workers: int,
    projection_only: bool,
    yes: bool,
    session_id: str | None,
    fresh_worktrees: bool,
) -> None:
    def _action() -> None:
        print("[RUN] ic optimize")
        print(
            "[FLOW] evaluate current IC behavior -> build writer feedback -> generate candidate policy update in isolated workspaces -> run pipeline correctness checks -> accept or reject the best update"
        )
        (
            project_dataset,
            worktree_manager,
            session_cfg,
            selected_tasks,
            selection_info,
            projection,
            should_exit,
        ) = _prepare_flow_run(
            command="ic optimize",
            max_items=max_items,
            offset=offset,
            max_workers=max_workers,
            yes=yes,
            projection_only=projection_only,
            session_id=session_id,
            fresh_worktrees=fresh_worktrees,
        )
        if should_exit:
            return
        materialized_tasks = _materialize_tasks_for_optimization(selected_tasks, worktree_manager)
        total = len(materialized_tasks)
        effective_workers = max(1, min(max_workers, total or 1))
        session_id_value = session_cfg.session_id or "ic-optimize"
        workspace_base = _optimize_workspace_base(session_id_value)
        workspace_base.mkdir(parents=True, exist_ok=True)
        print(
            "[RUN] ic optimize session=%s dataset=%s selected=%s execution=parallel effective_workers=%s isolation=per-task-workspace workspace_base=%s"
            % (
                session_id_value,
                project_dataset.name,
                total,
                effective_workers,
                workspace_base,
            )
        )

        started = 0
        completed = 0
        failed = 0
        progress_lock = threading.Lock()
        print_lock = threading.Lock()
        results: list[OptimizeTaskResult] = []

        def log(message: str) -> None:
            with print_lock:
                print(message)

        def run_one(task: LCATask, ordinal: int, parent_span: object) -> OptimizeTaskResult:
            nonlocal started, completed, failed
            started_at = time.perf_counter()
            with progress_lock:
                started += 1
                local_started = started
                local_completed = completed
                local_failed = failed
            log(
                "[TASK] ic optimize start task=%s/%s identity=%s repo=%s"
                % (ordinal, total, task.task_id, task.repo or "unknown")
            )
            _print_optimize_progress(
                started=local_started,
                completed=local_completed,
                failed=local_failed,
                total=total,
            )
            workspace_root: Path | None = None
            isolated_task = task
            try:
                workspace_root, isolated_task = _prepare_isolated_optimization_workspace(
                    task,
                    workspace_base=workspace_base,
                    ordinal=ordinal,
                )
                log(
                    "[TASK] ic optimize materialized task=%s/%s identity=%s worktree=%s optimize_workspace=%s"
                    % (
                        ordinal,
                        total,
                        task.task_id,
                        isolated_task.repo or task.repo or "unknown",
                        workspace_root,
                    )
                )
                with start_observation(
                    name="optimize_task",
                    parent=parent_span,
                    metadata={
                        "identity": task.task_id,
                        "ordinal": ordinal,
                        "total": total,
                        "repo": task.repo,
                        "isolated_repo": isolated_task.repo,
                        "workspace_root": str(workspace_root),
                    },
                ) as task_span:
                    log(
                        "[TASK] ic optimize phase=evaluate_policy task=%s/%s identity=%s"
                        % (ordinal, total, task.task_id)
                    )
                    with isolated_optimization_workspace(workspace_root):
                        history = run_loop(
                            isolated_task,
                            parent_trace=task_span,
                            session_id=session_id_value,
                        )
                    duration = time.perf_counter() - started_at
                    accepted = any(
                        getattr(record, "accepted_policy", None) is not None
                        for record in history
                    )
                    try:
                        task_span.update(
                            metadata={
                                "status": "success",
                                "iterations": len(history),
                                "accepted_policy": accepted,
                                "duration_seconds": duration,
                            }
                        )
                    except Exception:
                        pass
                with progress_lock:
                    completed += 1
                    local_started = started
                    local_completed = completed
                    local_failed = failed
                log(
                    "[TASK] ic optimize complete task=%s/%s identity=%s status=success iterations=%s accepted_policy=%s duration=%.2fs"
                    % (ordinal, total, task.task_id, len(history), accepted, duration)
                )
                _print_optimize_progress(
                    started=local_started,
                    completed=local_completed,
                    failed=local_failed,
                    total=total,
                )
                return OptimizeTaskResult(
                    task=task,
                    ordinal=ordinal,
                    total=total,
                    status="success",
                    history=history,
                    duration_seconds=duration,
                    workspace_root=workspace_root,
                    isolated_repo=isolated_task.repo,
                )
            except Exception as exc:  # noqa: BLE001
                duration = time.perf_counter() - started_at
                with start_observation(
                    name="optimize_task",
                    parent=parent_span,
                    metadata={
                        "identity": task.task_id,
                        "ordinal": ordinal,
                        "total": total,
                        "repo": task.repo,
                        "workspace_root": str(workspace_root) if workspace_root else None,
                        "status": "failure",
                        "error": str(exc),
                        "duration_seconds": duration,
                    },
                ):
                    pass
                with progress_lock:
                    failed += 1
                    local_started = started
                    local_completed = completed
                    local_failed = failed
                log(
                    "[TASK] ic optimize complete task=%s/%s identity=%s status=failure duration=%.2fs message=%s"
                    % (ordinal, total, task.task_id, duration, exc)
                )
                _print_optimize_progress(
                    started=local_started,
                    completed=local_completed,
                    failed=local_failed,
                    total=total,
                )
                return OptimizeTaskResult(
                    task=task,
                    ordinal=ordinal,
                    total=total,
                    status="failure",
                    history=[],
                    duration_seconds=duration,
                    workspace_root=workspace_root,
                    isolated_repo=isolated_task.repo,
                    error=str(exc),
                )

        with propagate_context(
            trace_name="ic_optimize",
            session_id=session_id_value,
            metadata={
                "command": "ic optimize",
                "dataset": project_dataset.name,
                "dataset_version": project_dataset.version,
                "selection": selection_info,
                "projection": projection,
                "execution": "parallel",
                "effective_workers": effective_workers,
                "isolation": "per-task-workspace",
            },
        ):
            with start_observation(
                name="ic_optimize_dataset",
                metadata={
                    "command": "ic optimize",
                    "dataset": project_dataset.name,
                    "dataset_version": project_dataset.version,
                    "selected_count": total,
                    "execution": "parallel",
                    "effective_workers": effective_workers,
                    "workspace_base": str(workspace_base),
                },
            ) as dataset_span:
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    futures = [
                        executor.submit(run_one, task, index, dataset_span)
                        for index, task in enumerate(materialized_tasks, start=1)
                    ]
                    for future in as_completed(futures):
                        results.append(future.result())
                applied, applied_task = _apply_best_accepted_policy(results, dataset_span)
                try:
                    dataset_span.update(
                        metadata={
                            "started": started,
                            "completed": completed,
                            "failed": failed,
                            "remaining": max(0, total - completed - failed),
                            "policy_applied": applied,
                            "applied_task": applied_task,
                        }
                    )
                except Exception:
                    pass
        total_duration = sum(result.duration_seconds for result in results)
        print(
            "[RUN] ic optimize summary selected=%s started=%s completed=%s failed=%s effective_workers=%s policy_applied=%s applied_task=%s total_task_duration=%.2fs"
            % (
                total,
                started,
                completed,
                failed,
                effective_workers,
                applied,
                applied_task or "none",
                total_duration,
            )
        )
        if failed:
            print(
                "[RUN] ic optimize failures=%s; successful task workspaces remain under %s for inspection"
                % (failed, workspace_base)
            )

    _run_with_langfuse_finally(_action)


def _run_sync_hf_command(
    *,
    dataset: str | None,
    config: str | None,
    split: str | None,
    hf_dataset: str,
    revision: str | None,
    token: str | None,
) -> None:
    def _action() -> None:
        ensure_langfuse_auth()
        project_dataset = resolve_project_dataset()
        target_dataset = dataset or project_dataset.name
        dataset_config = config or project_dataset.hf_config
        dataset_split = split or project_dataset.hf_split
        dataset_revision = revision if revision is not None else project_dataset.hf_revision
        sync_result = sync_hf_localization_dataset_to_langfuse(
            hf_dataset=hf_dataset,
            langfuse_dataset=target_dataset,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            revision=dataset_revision,
            token=token,
        )
        print(
            "[SYNC] HF dataset=%s config=%s split=%s revision=%s -> Langfuse dataset=%s source=%s existing=%s created=%s skipped=%s"
            % (
                hf_dataset,
                dataset_config,
                dataset_split,
                dataset_revision,
                target_dataset,
                sync_result.source_items,
                sync_result.existing_items,
                sync_result.created_items,
                sync_result.skipped_items,
            )
        )

    _run_with_langfuse_finally(_action)


@jc_app.command("run", help=flow_help(JC_RUN_NARRATIVE))
def jc_run(
    max_items: MaxItemsOption = None,
    offset: OffsetOption = 0,
    max_workers: MaxWorkersOption = 1,
    projection_only: ProjectionOnlyOption = False,
    yes: YesOption = False,
    session_id: SessionIdOption = None,
    fresh_worktrees: FreshWorktreesOption = False,
) -> None:
    _run_localization_flow_command(
        command_name="jc run",
        backend="jc",
        max_items=max_items,
        offset=offset,
        max_workers=max_workers,
        projection_only=projection_only,
        yes=yes,
        session_id=session_id,
        fresh_worktrees=fresh_worktrees,
    )


@ic_app.command("run", help=flow_help(IC_RUN_NARRATIVE))
def ic_run(
    max_items: MaxItemsOption = None,
    offset: OffsetOption = 0,
    max_workers: MaxWorkersOption = 8,
    projection_only: ProjectionOnlyOption = False,
    yes: YesOption = False,
    session_id: SessionIdOption = None,
    fresh_worktrees: FreshWorktreesOption = False,
) -> None:
    _run_localization_flow_command(
        command_name="ic run",
        backend="ic",
        max_items=max_items,
        offset=offset,
        max_workers=max_workers,
        projection_only=projection_only,
        yes=yes,
        session_id=session_id,
        fresh_worktrees=fresh_worktrees,
    )


@ic_app.command("optimize", help=flow_help(IC_OPTIMIZE_NARRATIVE))
def ic_optimize(
    max_items: MaxItemsOption = None,
    offset: OffsetOption = 0,
    max_workers: OptimizeMaxWorkersOption = 1,
    projection_only: ProjectionOnlyOption = False,
    yes: YesOption = False,
    session_id: SessionIdOption = None,
    fresh_worktrees: FreshWorktreesOption = False,
) -> None:
    _run_ic_optimize_command(
        max_items=max_items,
        offset=offset,
        max_workers=max_workers,
        projection_only=projection_only,
        yes=yes,
        session_id=session_id,
        fresh_worktrees=fresh_worktrees,
    )


@app.command("sync-hf")
def sync_hf(
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Override target Langfuse dataset"),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option("--config", help="Override HF dataset config"),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option("--split", help="Override HF dataset split"),
    ] = None,
    hf_dataset: Annotated[
        str,
        typer.Option("--hf-dataset", help="HuggingFace dataset to ingest"),
    ] = HF_DATASET_NAME,
    revision: Annotated[
        str | None,
        typer.Option("--revision", help="Override HF dataset revision pin"),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option("--token", help="Optional HuggingFace token"),
    ] = None,
) -> None:
    _run_sync_hf_command(
        dataset=dataset,
        config=config,
        split=split,
        hf_dataset=hf_dataset,
        revision=revision,
        token=token,
    )


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("LANGFUSE_STRICT_DEBUG", "1")
    logging.basicConfig(level=logging.INFO)
    args = argv if argv is not None else sys.argv[1:]
    _reject_removed_flags(args)
    app(args=args, standalone_mode=False)


if __name__ == "__main__":
    main()
