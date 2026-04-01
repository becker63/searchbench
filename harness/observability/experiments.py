from __future__ import annotations

from typing import Mapping, Sequence

from ..loop import run_loop
from ..loop_types import iteration_record_to_public_dict
from ..scorer import score
from ..utils.repo_targets import resolve_repo_target
from .baselines import (
    BaselineBundle,
    BaselineSnapshot,
    compute_baseline_for_item,
    index_baselines,
    make_baseline_bundle,
    require_baseline,
    resolve_baseline,
)
from .datasets import DatasetItem, fetch_dataset_items, normalize_dataset_item
from .langfuse import flush_langfuse
from .runner_integration import HostedRunResult, run_hosted_dataset_experiment
from .score_emitter import emit_score_for_handle


def _build_bundle_from_hosted(
    run: HostedRunResult, snapshots: list[BaselineSnapshot]
) -> BaselineBundle:
    return make_baseline_bundle(
        snapshots,
        dataset_name=run.dataset_name,
        dataset_version=run.dataset_version,
        baseline_run_id=run.run_id,
        baseline_run_name=run.run_name,
        dataset_run_id=run.run_id,
        metadata={"run_kind": "jc_baseline", "run_name": run.run_name},
    )


def run_hosted_jc_baseline_experiment(
    name: str, version: str | None = None
) -> BaselineBundle:
    def _run_item(
        item: Mapping[str, object], parent_trace: object | None = None
    ) -> Mapping[str, object]:
        dataset_item = normalize_dataset_item(item)
        snapshot = compute_baseline_for_item(
            dataset_item,
            dataset_name=name,
            dataset_version=version,
            parent_trace=parent_trace,
        )
        return {"baseline": snapshot.model_dump()}

    hosted = run_hosted_dataset_experiment(
        dataset_name=name,
        dataset_version=version,
        experiment_name="jc_baseline",
        task_fn=_run_item,
        metadata={"run_kind": "jc_baseline"},
    )
    snapshots: list[BaselineSnapshot] = []
    for entry in hosted.items:
        if entry.output and "baseline" in entry.output:
            try:
                snapshots.append(
                    BaselineSnapshot.model_validate(entry.output["baseline"])
                )
            except Exception:
                continue
    if not snapshots:
        raise RuntimeError("Hosted baseline experiment returned no results")
    flush_langfuse()
    return _build_bundle_from_hosted(hosted, snapshots)


def _run_ic_optimizations(
    items: Sequence[DatasetItem],
    baselines: BaselineBundle,
    dataset_name: str | None,
    dataset_version: str | None,
    iterations: int,
    hosted: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    indexed = index_baselines(baselines)

    def _run_item(
        item: Mapping[str, object], parent_trace: object | None = None
    ) -> Mapping[str, object]:
        normalized_item = normalize_dataset_item(item)
        resolved_repo = resolve_repo_target(normalized_item.repo)
        baseline = resolve_baseline(
            list(indexed.values()), normalized_item
        ) or require_baseline(baselines, normalized_item)
        task = normalized_item.model_dump()
        task["repo"] = resolved_repo
        history = run_loop(
            task,
            iterations=iterations,
            parent_trace=parent_trace,
            baseline_snapshot=baseline,
        )
        last_record = history[-1] if history else None
        last_dict: Mapping[str, object] = (
            iteration_record_to_public_dict(last_record) if last_record else {}
        )
        metrics = score(
            {"iterative_context": last_dict, "jcodemunch": baseline.jc_result}
        )
        for name, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                emit_score_for_handle(
                    parent_trace,
                    name=f"ic_opt.{name}",
                    value=float(value),
                    data_type="BOOLEAN" if isinstance(value, bool) else "NUMERIC",
                )
        history_dicts = [iteration_record_to_public_dict(r) for r in history]
        return {
            "item": normalized_item.model_dump(),
            "baseline": baseline.model_dump(),
            "history": history_dicts,
            "metrics": metrics,
        }

    hosted_result = run_hosted_dataset_experiment(
        dataset_name=dataset_name or "",
        dataset_version=dataset_version,
        experiment_name="ic_optimization",
        task_fn=_run_item,
        metadata={
            "run_kind": "ic_optimization",
            "baseline_run_id": baselines.baseline_run_id,
        },
    )
    for entry in hosted_result.items:
        if entry.output:
            results.append(dict(entry.output))
    if not results:
        raise RuntimeError("Hosted IC optimization returned no results")
    flush_langfuse()
    return results


def run_hosted_ic_optimization_experiment(
    name: str,
    version: str | None = None,
    iterations: int = 5,
    baselines: BaselineBundle | None = None,
    recompute_baselines: bool = False,
) -> list[dict[str, object]]:
    items = fetch_dataset_items(name=name, version=version)
    if baselines is None and not recompute_baselines:
        raise ValueError(
            "Baselines bundle required for optimization; set recompute_baselines=True to compute."
        )
    bundle = (
        baselines
        if baselines is not None
        else run_hosted_jc_baseline_experiment(name, version=version)
    )
    return _run_ic_optimizations(
        items,
        bundle,
        dataset_name=name,
        dataset_version=version,
        iterations=iterations,
        hosted=True,
    )
