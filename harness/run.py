from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from harness.loop import run_loop
from harness.observability.baselines import BaselineBundle, load_baseline_bundle, save_baseline_bundle
from harness.observability.experiments import (
    run_hosted_ic_optimization_experiment,
    run_hosted_jc_baseline_experiment,
)
from harness.observability.langfuse import flush_langfuse
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
    args = parser.parse_args(argv)

    try:
        if args.dataset and args.jc_baseline:
            print(f"[RUN] Hosted JC baseline experiment: {args.dataset}")
            bundle = run_hosted_jc_baseline_experiment(args.dataset, version=args.version)
            if args.baseline_path:
                save_baseline_bundle(bundle, Path(args.baseline_path))
                print(f"[RUN] Baseline bundle saved to {args.baseline_path}")
            print(f"[RUN] Baselines computed: {len(bundle.get('items', []))}")
        elif args.dataset and args.ic_optimize:
            print(f"[RUN] Hosted IC optimization experiment: {args.dataset}")
            bundle = _load_baseline(args.baseline_path)
            if bundle:
                print("[RUN] Loaded baseline bundle")
            results = run_hosted_ic_optimization_experiment(
                args.dataset,
                version=args.version,
                iterations=args.iterations,
                baselines=bundle,
                recompute_baselines=args.recompute_baselines,
            )
            print(f"[RUN] Optimization items completed: {len(results)}")
        else:
            repo_ref = args.repo or "small"
            resolved_repo = resolve_repo_target(repo_ref)
            task: Mapping[str, object] = {"symbol": args.symbol, "repo": resolved_repo}
            print(f"[RUN] Ad hoc run for repo '{repo_ref}' -> {resolved_repo}, symbol='{args.symbol}'")
            history = run_loop(task, iterations=args.iterations)
            last = history[-1] if history else {}
            last_score: Any = last.get("score") if isinstance(last, Mapping) else None
            print(f"[RUN] Completed {len(history)} iterations. Last score: {last_score}")
            if isinstance(last, Mapping):
                failed_step = last.get("failed_step")
                failed_exit = last.get("failed_exit_code")
                failed_summary = last.get("failed_summary")
                last_error = last.get("error") or last.get("writer_error")
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
