from __future__ import annotations

from typing import Mapping, Sequence

from .baselines import (
    BaselineBundle,
    BaselineSnapshot,
    compute_baseline_for_item,
    index_baselines,
    make_baseline_bundle,
    require_baseline,
    resolve_baseline,
)
from .datasets import Dataset, DatasetItem, fetch_dataset_items, normalize_dataset_item
from .langfuse import flush_langfuse, record_score, start_trace
from .runner_integration import HostedRunResult, run_hosted_dataset_experiment, run_local_dataset_experiment
from ..loop import run_loop
from ..loop_types import iteration_record_to_public_dict
from ..scorer import score
from ..utils.repo_targets import resolve_repo_target


def _build_bundle_from_hosted(run: HostedRunResult, snapshots: list[BaselineSnapshot]) -> BaselineBundle:
    return make_baseline_bundle(
        snapshots,
        dataset_name=run.dataset_name,
        dataset_version=run.dataset_version,
        baseline_run_id=run.run_id,
        baseline_run_name=run.run_name,
        dataset_run_id=run.run_id,
        metadata={"run_kind": "jc_baseline", "run_name": run.run_name},
    )


def _run_local_jc_baselines(items: Sequence[DatasetItem], dataset_name: str | None, dataset_version: str | None) -> BaselineBundle:
    def _run_item(item: Mapping[str, object], parent_trace: object | None = None) -> Mapping[str, object]:
        dataset_item = normalize_dataset_item(item)
        snapshot = compute_baseline_for_item(
            dataset_item,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            parent_trace=parent_trace,
        )
        return {"baseline": snapshot.model_dump()}

    hosted = run_local_dataset_experiment(
        data=[it.model_dump() for it in items],
        experiment_name="jc_baseline_local",
        task_fn=_run_item,
        metadata={"run_kind": "jc_baseline"},
    )
    baselines: list[BaselineSnapshot] = []
    for entry in hosted.items:
        if entry.output and "baseline" in entry.output:
            try:
                baselines.append(BaselineSnapshot.model_validate(entry.output["baseline"]))
            except Exception:
                continue
    return make_baseline_bundle(
        baselines,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        baseline_run_id=hosted.run_id,
        baseline_run_name=hosted.run_name,
        metadata={"run_kind": "jc_baseline"},
    )


def run_hosted_jc_baseline_experiment(name: str, version: str | None = None) -> BaselineBundle:
    def _run_item(item: Mapping[str, object], parent_trace: object | None = None) -> Mapping[str, object]:
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
                snapshots.append(BaselineSnapshot.model_validate(entry.output["baseline"]))
            except Exception:
                continue
    if not snapshots:
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
    items_raw = dataset.get("items")
    items_seq = items_raw if isinstance(items_raw, Sequence) else []
    filtered_items = [normalize_dataset_item(entry) for entry in items_seq if isinstance(entry, Mapping)]
    name_val = dataset.get("name") if isinstance(dataset, Mapping) else "local"
    ds = Dataset(name=str(name_val or "local"), items=filtered_items)
    ds_items: list[DatasetItem] = list(ds.entries)
    return _run_local_jc_baselines(ds_items, dataset_name=ds.name, dataset_version=None)


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

    def _run_item(item: Mapping[str, object], parent_trace: object | None = None) -> Mapping[str, object]:
        normalized_item = normalize_dataset_item(item)
        resolved_repo = resolve_repo_target(normalized_item.repo)
        baseline = resolve_baseline(list(indexed.values()), normalized_item) or require_baseline(baselines, normalized_item)
        task = normalized_item.model_dump()
        task["repo"] = resolved_repo
        history = run_loop(task, iterations=iterations, parent_trace=parent_trace, baseline_snapshot=baseline)
        last_record = history[-1] if history else None
        last_dict: Mapping[str, object] = (
            iteration_record_to_public_dict(last_record) if last_record else {}
        )
        metrics = score({"iterative_context": last_dict, "jcodemunch": baseline.jc_result})
        for name, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                record_score(parent_trace, f"ic_opt.{name}", float(value))
        history_dicts = [iteration_record_to_public_dict(r) for r in history]
        return {
            "item": normalized_item.model_dump(),
            "baseline": baseline.model_dump(),
            "history": history_dicts,
            "metrics": metrics,
        }

    if hosted:
        hosted_result = run_hosted_dataset_experiment(
            dataset_name=dataset_name or "",
            dataset_version=dataset_version,
            experiment_name="ic_optimization",
            task_fn=_run_item,
            metadata={"run_kind": "ic_optimization", "baseline_run_id": baselines.baseline_run_id},
        )
        for entry in hosted_result.items:
            if entry.output:
                results.append(dict(entry.output))
    else:
        local_result = run_local_dataset_experiment(
            data=[item.model_dump() for item in items],
            experiment_name="ic_optimization_local",
            task_fn=_run_item,
            metadata={"run_kind": "ic_optimization", "baseline_run_id": baselines.baseline_run_id},
        )
        for entry in local_result.items:
            if entry.output:
                results.append(dict(entry.output))
    if not results:
        root = start_trace(
            "ic_optimization_experiment",
            metadata={"dataset": dataset_name, "version": dataset_version, "count": len(items)},
        )
        for item in items:
            resolved_repo = resolve_repo_target(item.repo)
            baseline = resolve_baseline(list(indexed.values()), item) or require_baseline(baselines, item)
            task = item.model_dump()
            task["repo"] = resolved_repo
            history = run_loop(task, iterations=iterations, parent_trace=root, baseline_snapshot=baseline)
            history_entries: list[Mapping[str, object]] = [
                iteration_record_to_public_dict(r) for r in history
            ]
            results.append({"item": item.model_dump(), "baseline": baseline.model_dump(), "history": history_entries})
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
    if isinstance(dataset, Dataset):
        ds = dataset
    else:
        items_raw = dataset.get("items")
        items_seq = items_raw if isinstance(items_raw, Sequence) else []
        filtered_items = [normalize_dataset_item(entry) for entry in items_seq if isinstance(entry, Mapping)]
        name_val = dataset.get("name")
        ds = Dataset(name=str(name_val or "local"), items=filtered_items)
    ds_items: list[DatasetItem] = list(ds.entries)
    bundle = baselines if baselines is not None else _run_local_jc_baselines(ds_items, dataset_name=ds.name, dataset_version=None)
    return _run_ic_optimizations(ds_items, bundle, dataset_name=ds.name, dataset_version=None, iterations=iterations, hosted=False)
