from __future__ import annotations

"""
Localization-first CLI entrypoint: parse arguments, build typed requests, and run LCA bug localization flows.
Adds dataset guardrails: deterministic selection, cost projection, confirmation gating, and automation safety.
"""

import argparse
import logging
import sys
import os
from contextlib import contextmanager
from typing import IO, Any, Iterable, Iterator

try:
    import termios  # type: ignore
    import tty  # type: ignore
except ImportError:
    tty = None  # type: ignore
    termios = None  # type: ignore

from harness.localization.runtime.evaluate import (
    evaluate_localization_batch as _evaluate_localization_batch,
)
from harness.localization.materialization.hf_materialize import HuggingFaceRepoMaterializer
from harness.localization.models import LCATask
from harness.telemetry.tracing.cerebras_pricing import cost_details_for_usage
from harness.telemetry.hosted.experiments import (
    run_hosted_localization_baseline,
    run_hosted_localization_experiment,
)
from harness.telemetry.hosted.hf_lca import (
    HFDatasetLoadError,
    fetch_hf_localization_dataset,
)
from harness.telemetry.tracing import flush_langfuse, ensure_langfuse_auth
from harness.entrypoints.models.requests import (
    HostedLocalizationBaselineRequest,
    HostedLocalizationRunRequest,
)
from harness.telemetry.tracing.session_policy import SessionConfig

WARNING_THRESHOLD = 50
HF_DATASET_NAME = "JetBrains-Research/lca-bug-localization"
HF_DATASET_SOURCE = "huggingface"
REMOVED_FLAGS = {
    "--mode",
    "--dataset",
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


def evaluate_localization_batch(**kwargs: object):
    """
    Thin wrapper to keep monkeypatch targets stable for tests while deferring import cycles.
    """
    return _evaluate_localization_batch(**kwargs)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    # Default to real CLI arguments when caller does not supply an override.
    argv = argv if argv is not None else sys.argv[1:]
    for flag in REMOVED_FLAGS:
        if flag in argv:
            raise SystemExit(
                f"{flag} is no longer supported; use HF baseline/experiment subcommands with --config/--split/--revision/--max-items/--offset."
            )
    parser = argparse.ArgumentParser(
        description="HF LCA localization runner (baseline/experiment)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--config", required=True, help="HF dataset config (e.g., py/java/kt)"
        )
        sub.add_argument(
            "--split", required=True, help="HF dataset split (e.g., dev/test/train)"
        )
        sub.add_argument("--revision", help="HF dataset revision pin")
        sub.add_argument(
            "--max-items", type=int, help="Maximum number of dataset items to evaluate"
        )
        sub.add_argument(
            "--offset",
            type=int,
            default=0,
            help="Offset into the dataset before selection window",
        )
        sub.add_argument(
            "--max-workers",
            type=int,
            default=1,
            help="Maximum parallel workers for outer dataset task execution (>=1, applied after selection)",
        )
        sub.add_argument(
            "--projection-only",
            action="store_true",
            help="Compute selection and cost projection, then exit after confirmation without executing runs",
        )
        sub.add_argument(
            "--yes",
            dest="yes",
            action="store_true",
            help="Bypass interactive confirmation (requires --max-items)",
        )
        sub.add_argument("--session-id", help="Optional explicit session id")
        sub.add_argument(
            "--allow-session-fallback",
            action="store_true",
            default=True,
            help="Allow deterministic fallback session id generation at entrypoints",
        )

    _add_common(subparsers.add_parser("baseline"))
    _add_common(subparsers.add_parser("experiment"))

    return parser.parse_args(argv)


def _print_mat_summary(prefix: str, identity: Any, materialization: Any) -> None:
    if materialization:
        print(f"[MAT] {prefix} {identity}: {materialization.summary()}")


def _fmt_identity(identity: Any) -> str:
    if isinstance(identity, dict):
        return str(
            identity.get("id")
            or identity.get("task_id")
            or identity.get("repo")
            or identity
        )
    return str(identity)


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
    dataset_config: str | None,
    dataset_split: str | None,
    source: str,
    selection: dict[str, object],
) -> None:
    limit = selection.get("limit")
    offset = selection.get("offset")
    selected_count = selection.get("selected_count")
    total_available = selection.get("total_available")
    preview = selection.get("preview") or []
    print(
        "[SELECT] dataset=%s config=%s split=%s source=%s offset=%s limit=%s selected=%s total=%s"
        % (
            dataset,
            dataset_config,
            dataset_split,
            source,
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


def main(argv: list[str] | None = None) -> None:
    # Enable strict Langfuse debug behavior during migration/debugging.
    os.environ.setdefault("LANGFUSE_STRICT_DEBUG", "1")
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    # Fail fast on Langfuse auth/base-url issues in the canonical path.
    ensure_langfuse_auth()
    if args.max_workers is not None and args.max_workers <= 0:
        raise ValueError("--max-workers must be a positive integer")
    materializer = HuggingFaceRepoMaterializer()
    session_cfg = SessionConfig(
        session_id=args.session_id,
        session_scope="run",  # dataset-backed runs use run scope
        allow_generated_fallback=args.allow_session_fallback,
    )

    try:
        if args.command == "baseline":
            print(f"[RUN] HF localization baseline for dataset '{HF_DATASET_NAME}'")
            tasks = fetch_hf_localization_dataset(
                HF_DATASET_NAME,
                dataset_config=args.config,
                dataset_split=args.split,
                revision=args.revision,
            )
            if not tasks:
                raise RuntimeError("No localization tasks available for baseline run")
            selected_tasks, selection_info = _select_tasks(
                tasks, args.max_items, args.offset
            )
            _warn_if_unbounded(selection_info, args.max_items)
            _print_selection(
                HF_DATASET_NAME,
                args.config,
                args.split,
                HF_DATASET_SOURCE,
                selection_info,
            )
            projection = _compute_projection(
                _coerce_int(selection_info.get("selected_count"))
            )
            _print_projection(selection_info, projection)
            if args.yes and not args.max_items:
                raise ValueError("--yes requires --max-items to be set for safety.")
            _require_confirmation(args.yes, selection_info, projection)
            if args.projection_only:
                print("[RUN] Projection-only mode: exiting before execution.")
                return
            req = HostedLocalizationBaselineRequest(
                dataset=HF_DATASET_NAME,
                version=args.revision,
                dataset_config=args.config,
                dataset_split=args.split,
                session=session_cfg,
                max_items=args.max_items,
                offset=args.offset,
                max_workers=args.max_workers,
                selection=selection_info,
                projection=projection,
            )
            bundle = run_hosted_localization_baseline(
                req, materializer=materializer, tasks=selected_tasks
            )
            if bundle is None:
                print("[RUN] Baseline run returned no bundle; skipping summary.")
                return
            print(f"[RUN] Baselines computed: {len(bundle.items)}")
            failure = getattr(bundle, "failure", None)
            if failure is not None:
                category = getattr(failure.category, "value", failure.category)
                print(
                    f"[RUN] Baseline run reported failure category={category} task_id={failure.task_id} message={failure.message}"
                )
            for snap in bundle.items:
                _print_mat_summary("baseline", snap.identity, snap.materialization)
        elif args.command == "experiment":
            print(f"[RUN] HF localization experiment for dataset '{HF_DATASET_NAME}'")
            tasks = fetch_hf_localization_dataset(
                HF_DATASET_NAME,
                dataset_config=args.config,
                dataset_split=args.split,
                revision=args.revision,
            )
            if not tasks:
                raise RuntimeError("No localization tasks available for experiment run")
            selected_tasks, selection_info = _select_tasks(
                tasks, args.max_items, args.offset
            )
            _warn_if_unbounded(selection_info, args.max_items)
            _print_selection(
                HF_DATASET_NAME,
                args.config,
                args.split,
                HF_DATASET_SOURCE,
                selection_info,
            )
            projection = _compute_projection(
                _coerce_int(selection_info.get("selected_count"))
            )
            _print_projection(selection_info, projection)
            if args.yes and not args.max_items:
                raise ValueError("--yes requires --max-items to be set for safety.")
            _require_confirmation(args.yes, selection_info, projection)
            if args.projection_only:
                print("[RUN] Projection-only mode: exiting before execution.")
                return
            req = HostedLocalizationRunRequest(
                dataset=HF_DATASET_NAME,
                version=args.revision,
                dataset_config=args.config,
                dataset_split=args.split,
                session=session_cfg,
                max_items=args.max_items,
                offset=args.offset,
                max_workers=args.max_workers,
                selection=selection_info,
                projection=projection,
            )
            results = run_hosted_localization_experiment(
                req, materializer=materializer, tasks=selected_tasks
            )
            print(f"[RUN] Localization items completed: {len(results.items)}")
            failure = getattr(results, "failure", None)
            if failure is not None:
                category = getattr(failure.category, "value", failure.category)
                print(
                    f"[RUN] Experiment run reported failure category={category} task_id={failure.task_id} message={failure.message}"
                )
            for item in results.items:
                _print_mat_summary(
                    "experiment",
                    _fmt_identity(item.task.get("identity")),
                    item.materialization,
                )
    except HFDatasetLoadError as exc:
        print(f"[ERROR] HF dataset load failed (category={exc.category}): {exc}")
        raise
    except RuntimeError as exc:
        # Surface materialization failures or other runtime issues
        print(f"[ERROR] {exc}")
        raise
    finally:
        flush_langfuse()


if __name__ == "__main__":
    main()
