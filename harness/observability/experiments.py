from __future__ import annotations

from typing import Mapping, Sequence, cast

from .baselines import (
    BaselineBundle,
    BaselineSnapshot,
    compute_baseline_for_item,
    make_baseline_bundle,
    require_baseline,
    resolve_baseline,
)
from .datasets import DatasetItem, fetch_dataset_items, local_dataset, normalize_dataset_item
from .langfuse import flush_langfuse, record_score, start_trace
from .runner_integration import HostedRunResult, run_hosted_dataset_experiment, run_local_dataset_experiment
from ..utils.repo_targets import resolve_repo_target
from ..loop import run_loop
from ..scorer import score


def _build_bundle_from_hosted(run: HostedRunResult, snapshots: list[BaselineSnapshot]) -> BaselineBundle:
    return make_baseline_bundle(
        snapshots,
        dataset_name=run.get("dataset_name"),
        dataset_version=run.get("dataset_version"),
        baseline_run_id=run.get("run_id"),
        baseline_run_name=run.get("run_name"),
        dataset_run_id=run.get("run_id"),
        metadata={"run_kind": "jc_baseline", "run_name": run.get("run_name")},
    )


def _run_local_jc_baselines(items: Sequence[DatasetItem], dataset_name: str | None, dataset_version: str | None) -> BaselineBundle:
    def _run_item(item: Mapping[str, object], parent_trace: object | None = None) -> Mapping[str, object]:
        return {
            "baseline": compute_baseline_for_item(
                normalize_dataset_item(item), dataset_name=dataset_name, dataset_version=dataset_version, parent_trace=parent_trace
            )
        }

    hosted = run_local_dataset_experiment(
        data=list(items),
        experiment_name="jc_baseline_local",
        task_fn=_run_item,
        metadata={"run_kind": "jc_baseline"},
    )
    baselines: list[BaselineSnapshot] = []
    for entry in hosted.get("items", []):
        if not isinstance(entry, Mapping):
            continue
        output = entry.get("output")
        if isinstance(output, Mapping) and isinstance(output.get("baseline"), Mapping):
            baselines.append(cast(BaselineSnapshot, output["baseline"]))
    return make_baseline_bundle(
        baselines,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        baseline_run_id=hosted.get("run_id"),
        baseline_run_name=hosted.get("run_name"),
        metadata={"run_kind": "jc_baseline"},
    )


def run_hosted_jc_baseline_experiment(name: str, version: str | None = None) -> BaselineBundle:
    def _run_item(item: Mapping[str, object], parent_trace: object | None = None) -> Mapping[str, object]:
        snapshot = compute_baseline_for_item(
            normalize_dataset_item(item), dataset_name=name, dataset_version=version, parent_trace=parent_trace
        )
        return {"baseline": snapshot}

    hosted = run_hosted_dataset_experiment(
        dataset_name=name,
        dataset_version=version,
        experiment_name="jc_baseline",
        task_fn=_run_item,
        metadata={"run_kind": "jc_baseline"},
    )
    snapshots: list[BaselineSnapshot] = []
    for entry in hosted.get("items", []):
        if not isinstance(entry, Mapping):
            continue
        output = entry.get("output")
        if isinstance(output, Mapping) and "baseline" in output and isinstance(output["baseline"], Mapping):
            snapshots.append(cast(BaselineSnapshot, output["baseline"]))
    if not snapshots:
        # Fallback to manual computation if hosted wrapper returned nothing
        items = fetch_dataset_items(name=name, version=version)
        root = start_trace("jc_baseline_experiment", metadata={"dataset": name, "version": version, "count": len(items)})
        for item in items:
            snapshots.append(compute_baseline_for_item(item, dataset_name=name, dataset_version=version, parent_trace=root))
        if root and hasattr(root, "end"):
            try:
                root.end(metadata={"completed": len(snapshots)})
            except Exception:
                pass
        hosted = HostedRunResult(run_id=getattr(root, "id", None), run_name=None, dataset_name=name, dataset_version=version, items=[])
    flush_langfuse()
    return _build_bundle_from_hosted(hosted, snapshots)


def run_local_jc_baseline_experiment(dataset: Mapping[str, object]) -> BaselineBundle:
    items_obj = dataset.get("items") if isinstance(dataset, Mapping) else []
    items: list[DatasetItem] = []
    if isinstance(items_obj, Sequence):
        for entry in items_obj:
            if isinstance(entry, Mapping):
                items.append(normalize_dataset_item(entry))
    name = dataset.get("name") if isinstance(dataset, Mapping) else None
    dataset_name = name if isinstance(name, str) else None
    return _run_local_jc_baselines(items, dataset_name=dataset_name, dataset_version=None)


def _run_ic_optimizations(
    items: Sequence[DatasetItem],
    baselines: BaselineBundle,
    dataset_name: str | None,
    dataset_version: str | None,
    iterations: int,
    hosted: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    indexed = baselines.get("items", []) if isinstance(baselines, Mapping) else []

    def _run_item(item: Mapping[str, object], parent_trace: object | None = None) -> Mapping[str, object]:
        normalized_item = normalize_dataset_item(item)
        repo_ref = normalized_item.get("repo", "")
        resolved_repo = resolve_repo_target(str(repo_ref))
        baseline = resolve_baseline(indexed, normalized_item) if indexed else resolve_baseline(baselines, normalized_item)
        baseline = require_baseline(baselines, normalized_item) if baseline is None else baseline
        task = dict(normalized_item)
        task["repo"] = resolved_repo
        history = run_loop(task, iterations=iterations, parent_trace=parent_trace, baseline_snapshot=baseline)
        last = history[-1] if history else {}
        jc_payload: Mapping[str, object] | dict[str, object] = baseline["jc_result"] if baseline and "jc_result" in baseline else {}
        metrics = score({"iterative_context": last if isinstance(last, Mapping) else {}, "jcodemunch": jc_payload})
        for name, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                record_score(parent_trace, f"ic_opt.{name}", float(value))
        return {"item": normalized_item, "baseline": baseline, "history": history, "metrics": metrics}

    if hosted:
        hosted_result = run_hosted_dataset_experiment(
            dataset_name=dataset_name or "",
            dataset_version=dataset_version,
            experiment_name="ic_optimization",
            task_fn=_run_item,
            metadata={"run_kind": "ic_optimization", "baseline_run_id": baselines.get("baseline_run_id") if isinstance(baselines, Mapping) else None},
        )
        for entry in hosted_result.get("items", []):
            if not isinstance(entry, Mapping):
                continue
            output = entry.get("output")
            if isinstance(output, Mapping):
                results.append(dict(output))
    else:
        local_result = run_local_dataset_experiment(
            data=list(items),
            experiment_name="ic_optimization_local",
            task_fn=_run_item,
            metadata={"run_kind": "ic_optimization", "baseline_run_id": baselines.get("baseline_run_id") if isinstance(baselines, Mapping) else None},
        )
        for entry in local_result.get("items", []):
            if not isinstance(entry, Mapping):
                continue
            output = entry.get("output")
            if isinstance(output, Mapping):
                results.append(dict(output))
    if not results:
        # Fallback manual path if runner returned nothing
        root = start_trace(
            "ic_optimization_experiment",
            metadata={"dataset": dataset_name, "version": dataset_version, "count": len(items)},
        )
        for item in items:
            repo_ref = item.get("repo")
            resolved_repo = resolve_repo_target(str(repo_ref))
            baseline = resolve_baseline(indexed, item) if indexed else resolve_baseline(baselines, item)
            baseline = require_baseline(baselines, item) if baseline is None else baseline
            task = dict(item)
            task["repo"] = resolved_repo
            history = run_loop(task, iterations=iterations, parent_trace=root, baseline_snapshot=baseline)
            results.append({"item": item, "baseline": baseline, "history": history})
        if root and hasattr(root, "end"):
            try:
                root.end(metadata={"completed": len(results)})
            except Exception:
                pass
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
        raise ValueError("Baselines bundle required for optimization; set recompute_baselines=True to compute.")
    bundle = baselines if baselines is not None else run_hosted_jc_baseline_experiment(name, version=version)
    return _run_ic_optimizations(items, bundle, dataset_name=name, dataset_version=version, iterations=iterations, hosted=True)


def run_local_ic_optimization_experiment(
    dataset: Mapping[str, object],
    iterations: int = 5,
    baselines: BaselineBundle | None = None,
    recompute_baselines: bool = False,
) -> list[dict[str, object]]:
    items_source = dataset.get("items") if isinstance(dataset, Mapping) else []
    name = dataset.get("name") if isinstance(dataset, Mapping) else "local"
    normalized_ds = local_dataset(str(name), items_source if isinstance(items_source, Sequence) else [])
    items = normalized_ds.get("items", [])
    if not isinstance(items, list):
        items = []
    name_value = normalized_ds.get("name")
    dataset_name: str | None = name_value if isinstance(name_value, str) else None
    if baselines is None and not recompute_baselines:
        raise ValueError("Baselines bundle required for optimization; set recompute_baselines=True to compute.")
    bundle = baselines if baselines is not None else _run_local_jc_baselines(items, dataset_name=dataset_name, dataset_version=None)
    return _run_ic_optimizations(items, bundle, dataset_name=dataset_name, dataset_version=None, iterations=iterations, hosted=False)


__all__ = [
    "run_hosted_jc_baseline_experiment",
    "run_local_jc_baseline_experiment",
    "run_hosted_ic_optimization_experiment",
    "run_local_ic_optimization_experiment",
]
