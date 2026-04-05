from __future__ import annotations

"""
Localization-first CLI entrypoint: parse arguments, build typed requests, and run LCA bug localization flows.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from harness.localization.evaluation_backend import (
    LocalizationEvaluationRequest,
    evaluate_localization_batch as _evaluate_localization_batch,
)
from harness.localization.models import LCAContext, LCAGold, LCATaskIdentity
from harness.localization.hf_materializer import HuggingFaceRepoMaterializer
from harness.observability.baselines import load_baseline_bundle, save_baseline_bundle
from harness.observability.experiments import (
    run_hosted_localization_baseline,
    run_hosted_localization_experiment,
)
from harness.observability.requests import (
    HostedLocalizationBaselineRequest,
    HostedLocalizationRunRequest,
    LocalizationAdHocRequest,
    LocalizationDatasetSource,
)
from harness.observability.hf_lca import HFDatasetLoadError
from harness.observability.session_policy import SessionConfig, resolve_session_id
from harness.observability.langfuse import flush_langfuse


def evaluate_localization_batch(req: LocalizationEvaluationRequest):
    """
    Thin wrapper to keep monkeypatch targets stable for tests while deferring import cycles.
    """
    return _evaluate_localization_batch(req)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localization-native LCA runner")
    parser.add_argument(
        "--mode",
        choices=["localization-single", "localization-baseline", "localization-experiment"],
        default="localization-single",
        help="Localization mode: single task, dataset baseline, or dataset experiment",
    )
    parser.add_argument("--dataset", help="Localization dataset name (required for baseline/experiment)")
    parser.add_argument("--version", help="Dataset version")
    parser.add_argument("--config", dest="dataset_config", help="Dataset config (e.g., py/java/kt)")
    parser.add_argument("--split", dest="dataset_split", help="Dataset split (e.g., dev/train/test)")
    parser.add_argument(
        "--dataset-source",
        choices=[source.value for source in LocalizationDatasetSource],
        default=LocalizationDatasetSource.LANGFUSE.value,
        help="Dataset source: langfuse (default) or huggingface",
    )
    parser.add_argument("--repo-owner", required=False, help="Repository owner for LCA eval")
    parser.add_argument("--repo-name", required=False, help="Repository name for LCA eval")
    parser.add_argument("--base-sha", required=False, help="Base SHA (bug-reproducible snapshot) for LCA eval")
    parser.add_argument("--repo-path", required=False, help="Local path to repository to analyze (required without materializer)")
    parser.add_argument("--issue-url", help="Issue URL for LCA eval")
    parser.add_argument("--pull-url", help="Pull URL for LCA eval")
    parser.add_argument("--issue-title", help="Issue title for context", default="")
    parser.add_argument("--issue-body", help="Issue body for context", default="")
    parser.add_argument("--diff", help="Inline diff content", default=None)
    parser.add_argument("--diff-url", help="Diff URL", default=None)
    parser.add_argument("--changed-file", action="append", dest="changed_files", help="Gold changed file (repeatable)")
    parser.add_argument("--baseline-path", help="Path to load/save baseline bundle JSON")
    parser.add_argument(
        "--session-scope",
        choices=["run", "item"],
        help="Session scope (default run for single, item for dataset)",
    )
    parser.add_argument("--session-id", help="Optional explicit session id")
    parser.add_argument(
        "--allow-session-fallback",
        action="store_true",
        help="Allow deterministic fallback session id generation at entrypoints",
    )
    return parser.parse_args(argv)


def _print_mat_summary(prefix: str, identity: Any, materialization: Any) -> None:
    if materialization:
        print(f"[MAT] {prefix} {identity}: {materialization.summary()}")


def _fmt_identity(identity: Any) -> str:
    if isinstance(identity, dict):
        return str(identity.get("id") or identity.get("task_id") or identity.get("repo") or identity)
    return str(identity)


def _build_identity(args: argparse.Namespace) -> LCATaskIdentity:
    dataset_name = args.dataset or "adhoc-localization"
    dataset_config = args.dataset_config or "adhoc-config"
    dataset_split = args.dataset_split or "adhoc-split"
    repo_owner = args.repo_owner or "unknown-owner"
    repo_name = args.repo_name or "unknown-repo"
    base_sha = args.base_sha or "unknown-base-sha"
    return LCATaskIdentity(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        repo_owner=repo_owner,
        repo_name=repo_name,
        base_sha=base_sha,
        issue_url=args.issue_url,
        pull_url=args.pull_url,
    )


def _load_baseline(path: str | None):  # pyright: ignore[reportUnusedFunction]
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline bundle not found at {path}")
    return load_baseline_bundle(p)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    dataset_source = LocalizationDatasetSource(args.dataset_source)
    materializer = HuggingFaceRepoMaterializer() if dataset_source == LocalizationDatasetSource.HUGGINGFACE else None
    session_cfg = SessionConfig(
        session_id=args.session_id,
        session_scope=(args.session_scope or ("item" if args.mode != "localization-single" else "run")),  # type: ignore[arg-type]
        allow_generated_fallback=args.allow_session_fallback,
    )

    try:
        if args.mode == "localization-baseline":
            if not args.dataset:
                raise ValueError("--dataset is required for localization-baseline")
            print(f"[RUN] Hosted localization baseline for dataset '{args.dataset}'")
            req = HostedLocalizationBaselineRequest(
                dataset=args.dataset,
                version=args.version,
                dataset_config=args.dataset_config,
                dataset_split=args.dataset_split,
                session=session_cfg,
                dataset_source=dataset_source,
            )
            bundle = run_hosted_localization_baseline(req, materializer=materializer)
            if args.baseline_path:
                save_baseline_bundle(bundle, Path(args.baseline_path))
                print(f"[RUN] Baseline bundle saved to {args.baseline_path}")
            print(f"[RUN] Baselines computed: {len(bundle.items)}")
            if dataset_source == LocalizationDatasetSource.HUGGINGFACE:
                for snap in bundle.items:
                    _print_mat_summary("baseline", snap.identity, snap.materialization)
        elif args.mode == "localization-experiment":
            if not args.dataset:
                raise ValueError("--dataset is required for localization-experiment")
            print(f"[RUN] Hosted localization experiment for dataset '{args.dataset}'")
            req = HostedLocalizationRunRequest(
                dataset=args.dataset,
                version=args.version,
                dataset_config=args.dataset_config,
                dataset_split=args.dataset_split,
                session=session_cfg,
                dataset_source=dataset_source,
            )
            results = run_hosted_localization_experiment(req, materializer=materializer)
            print(f"[RUN] Localization items completed: {len(results.items)}")
            if dataset_source == LocalizationDatasetSource.HUGGINGFACE:
                for item in results.items:
                    _print_mat_summary("experiment", _fmt_identity(item.task.get("identity")), item.materialization)
        else:
            identity = _build_identity(args)
            gold = LCAGold(changed_files=args.changed_files or [])
            context = LCAContext(
                issue_title=args.issue_title,
                issue_body=args.issue_body,
                diff=args.diff,
                diff_url=args.diff_url,
                repo_language=None,
                repo_languages=[],
                repo_license=None,
                repo_stars=None,
            )
            req = LocalizationAdHocRequest(
                identity=identity,
                context=context,
                gold=gold,
                repo=args.repo_path,
                session=session_cfg,
                dataset_source=dataset_source,
            )
            resolved_session = resolve_session_id(session_cfg, run_id=identity.task_id())
            print(f"[RUN] Ad hoc localization run for {identity.repo_owner}/{identity.repo_name}@{identity.base_sha}")
            eval_result = evaluate_localization_batch(
                LocalizationEvaluationRequest(
                    tasks=[req.to_task()],
                    dataset_source=dataset_source.value,
                    materializer=materializer,
                    parent_trace=None,
                )
            )
            if eval_result.failure or not eval_result.items:
                category = getattr(eval_result.failure.category, "value", eval_result.failure.category) if eval_result.failure else "unknown"
                message = eval_result.failure.message if eval_result.failure else "evaluation_failed"
                raise RuntimeError(f"Localization evaluation failed ({category}): {message}")
            task_result = eval_result.items[0]
            print(f"[RUN] Prediction: {task_result.prediction}")
            print(f"[RUN] Metrics: {task_result.metrics.model_dump()}")
            print(f"[RUN] Session: {resolved_session}")
            if task_result.materialization:
                _print_mat_summary("single", identity.task_id, task_result.materialization)
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
