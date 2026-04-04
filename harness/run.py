from __future__ import annotations

"""
CLI entrypoint: parse arguments, build typed requests, and dispatch to application services.
No leaf logic or internal coordination lives here.
"""

import argparse
from pathlib import Path
from typing import Any, Mapping

from harness.observability.requests import (
    AdHocOptimizationRequest,
    HostedICOptimizationRequest,
    HostedJCBaselineRequest,
)
from harness.localization.eval import build_file_localization_eval_record
from harness.localization.materializer import RepoMaterializationRequest
from harness.localization.models import LCATaskIdentity, LCAPrediction, LCAGold
from harness.loop import IterationRecord, run_loop
from harness.observability.baselines import BaselineBundle, load_baseline_bundle, save_baseline_bundle
from harness.observability.experiments import (
    run_hosted_ic_optimization_experiment,
    run_hosted_jc_baseline_experiment,
)
from harness.observability.langfuse import flush_langfuse
from harness.observability.session_policy import SessionConfig, resolve_session_id
from harness.utils.repo_targets import resolve_repo_target


def _load_baseline(path: str | None) -> BaselineBundle | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline bundle not found at {path}")
    return load_baseline_bundle(p)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Searchbench Langfuse-first runner")
    parser.add_argument("--dataset", help="Langfuse dataset name")
    parser.add_argument("--version", help="Langfuse dataset version")
    parser.add_argument(
        "--mode",
        choices=["adhoc", "jc-baseline", "ic-optimize", "lca-bug-localization"],
        help="Runner mode",
    )
    parser.add_argument("--lca-config", help="LCA dataset config (py/java/kt)")
    parser.add_argument("--lca-split", help="LCA dataset split (dev/train/test)")
    parser.add_argument("--repo-owner", help="Repository owner for LCA eval")
    parser.add_argument("--repo-name", help="Repository name for LCA eval")
    parser.add_argument("--base-sha", help="Base SHA (bug-reproducible snapshot) for LCA eval")
    parser.add_argument("--issue-url", help="Issue URL for LCA eval")
    parser.add_argument("--pull-url", help="Pull URL for LCA eval")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--jc-baseline", action="store_true", help="Run JC baseline experiment (hosted dataset)")
    parser.add_argument("--ic-optimize", action="store_true", help="Run IC optimization experiment (hosted dataset)")
    parser.add_argument("--baseline-path", help="Path to load/save baseline bundle JSON")
    parser.add_argument("--recompute-baselines", action="store_true", help="Recompute baselines if none provided")
    parser.add_argument("--repo", help="Repo target ref or path for ad hoc run (e.g., small, medium)", default="small")
    parser.add_argument("--symbol", help="Symbol/function name for ad hoc run", default="expand_node")
    parser.add_argument("--session-id", help="Optional explicit session id")
    parser.add_argument(
        "--session-scope",
        choices=["run", "item"],
        help="Session scope (default run for ad hoc, item for hosted)",
    )
    parser.add_argument(
        "--allow-session-fallback",
        action="store_true",
        help="Allow deterministic fallback session id generation at entrypoints",
    )
    args = parser.parse_args(argv)

    try:
        mode = args.mode
        if args.dataset and args.jc_baseline:
            print(f"[RUN] Hosted JC baseline experiment: {args.dataset}")
            scope = args.session_scope or "item"
            session_cfg = SessionConfig(
                session_id=args.session_id,
                session_scope=scope,  # type: ignore[arg-type]
                allow_generated_fallback=args.allow_session_fallback,
            )
            jc_req = HostedJCBaselineRequest(dataset=args.dataset, version=args.version, session=session_cfg)
            bundle = run_hosted_jc_baseline_experiment(jc_req)
            if args.baseline_path:
                save_baseline_bundle(bundle, Path(args.baseline_path))
                print(f"[RUN] Baseline bundle saved to {args.baseline_path}")
            print(f"[RUN] Baselines computed: {len(bundle.items)}")
        elif args.dataset and args.ic_optimize:
            print(f"[RUN] Hosted IC optimization experiment: {args.dataset}")
            bundle = _load_baseline(args.baseline_path)
            if bundle:
                print("[RUN] Loaded baseline bundle")
            scope = args.session_scope or "item"
            session_cfg = SessionConfig(
                session_id=args.session_id,
                session_scope=scope,  # type: ignore[arg-type]
                allow_generated_fallback=args.allow_session_fallback,
            )
            ic_req = HostedICOptimizationRequest(
                dataset=args.dataset,
                version=args.version,
                iterations=args.iterations,
                baselines=bundle,
                recompute_baselines=args.recompute_baselines,
                session=session_cfg,
            )
            results = run_hosted_ic_optimization_experiment(ic_req)
            print(f"[RUN] Optimization items completed: {len(results)}")
        elif mode == "lca-bug-localization":
            _run_lca_stub(args)
        else:
            repo_ref = args.repo or "small"
            resolved_repo = resolve_repo_target(repo_ref)
            task: Mapping[str, object] = {"symbol": args.symbol, "repo": resolved_repo}
            print(f"[RUN] Ad hoc run for repo '{repo_ref}' -> {resolved_repo}, symbol='{args.symbol}'")
            session_cfg = SessionConfig(
                session_id=args.session_id,
                session_scope=(args.session_scope or "run"),  # type: ignore[arg-type]
                allow_generated_fallback=args.allow_session_fallback,
            )
            ad_req = AdHocOptimizationRequest(
                repo_ref=repo_ref,
                symbol=args.symbol,
                iterations=args.iterations,
                session=session_cfg,
            )
            resolved_session = resolve_session_id(
                session_cfg,
                run_id=repo_ref,
            )
            history = run_loop(task, iterations=ad_req.iterations, session_id=resolved_session)
            last: IterationRecord | None = history[-1] if history else None
            last_score: Any = last.metrics.get("score") if last else None
            print(f"[RUN] Completed {len(history)} iterations. Last score: {last_score}")
            if last:
                failed_step = last.failed_step
                failed_exit = last.failed_exit_code
                failed_summary = last.failed_summary
                last_error = last.error or last.writer_error
                if failed_step:
                    print(f"[RUN] Last failed step: {failed_step}")
                    if failed_exit is not None:
                        print(f"[RUN] Exit code: {failed_exit}")
                    if failed_summary:
                        print(f"[RUN] Failure summary: {failed_summary}")
                elif last_error:
                    print(f"[RUN] Last error: {last_error}")
    finally:
        flush_langfuse()


def _run_lca_stub(args: argparse.Namespace) -> None:
    """
    Phase-1 stub for LCA bug localization mode.
    No dataset loading or repo materialization is performed.
    Emits user-visible logs for repo preparation steps.
    """
    dataset = args.dataset or "unspecified-dataset"
    config = args.lca_config or "unspecified-config"
    split = args.lca_split or "unspecified-split"
    repo_owner = args.repo_owner or "unknown-owner"
    repo_name = args.repo_name or "unknown-repo"
    base_sha = args.base_sha or "unknown-base-sha"
    issue_url = args.issue_url
    pull_url = args.pull_url

    print("[LCA] Starting LCA bug-localization (stub mode, no execution)")
    print(f"[LCA] Dataset: name={dataset}, config={config}, split={split}")
    print(f"[LCA] Repo: {repo_owner}/{repo_name} @ base_sha={base_sha}")

    materialize_req = RepoMaterializationRequest(
        dataset_name=dataset,
        dataset_config=config,
        dataset_split=split,
        repo_owner=repo_owner,
        repo_name=repo_name,
        base_sha=base_sha,
    )

    # User-visible repo prep logs (stub; no fetch/extract/checkout performed)
    print("[LCA][repo] resolve repo target")
    print("[LCA][repo] cache lookup: not attempted (phase 1 stub)")
    print("[LCA][repo] download/fetch: not attempted (phase 1 stub)")
    print("[LCA][repo] extract/materialize: not attempted (phase 1 stub)")
    print("[LCA][repo] checkout base_sha: not attempted (phase 1 stub)")
    print("[LCA][repo] ready path: not available (phase 1 stub)")

    # Build identity and a placeholder eval record to show contract; no scoring executed.
    identity = LCATaskIdentity(
        dataset_name=materialize_req.dataset_name,
        dataset_config=materialize_req.dataset_config,
        dataset_split=materialize_req.dataset_split,
        repo_owner=materialize_req.repo_owner,
        repo_name=materialize_req.repo_name,
        base_sha=materialize_req.base_sha,
        issue_url=issue_url,
        pull_url=pull_url,
    )
    placeholder_pred = LCAPrediction(predicted_files=[])
    placeholder_gold = LCAGold(changed_files=[])
    record = build_file_localization_eval_record(identity, placeholder_pred, placeholder_gold, evidence=None, repo_path=None)
    print("[LCA] Canonical eval record shape (stub, empty data):")
    print(record)


if __name__ == "__main__":
    main()
