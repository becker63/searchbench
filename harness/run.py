from __future__ import annotations

"""
CLI entrypoint: parse arguments, build typed requests, and dispatch to application services.
No leaf logic or internal coordination lives here.
"""

import argparse
from pathlib import Path
from typing import Any, Mapping

from harness.entrypoints import (
    AdHocOptimizationRequest,
    HostedICOptimizationRequest,
    HostedJCBaselineRequest,
)
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


if __name__ == "__main__":
    main()
