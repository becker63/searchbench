from __future__ import annotations

"""
Application/experiment boundary: hosted JC baseline and IC optimization wrappers.
Accept normalized inputs or typed requests; do not embed CLI or leaf logic.
"""

import time
from typing import Any, Callable, Mapping, Sequence, cast

from .requests import HostedICOptimizationRequest, HostedJCBaselineRequest
from ..loop import iteration_record_to_public_dict, run_loop
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
from .langfuse import flush_langfuse, propagate_context
from .langfuse import get_langfuse_client
from pydantic import BaseModel, Field
from .session_policy import SessionConfig, resolve_session_id


class HostedRunItemResult(BaseModel):
    item: Mapping[str, object]
    output: Mapping[str, object] | dict[str, object] | None = None
    trace_id: str | None = None


class HostedRunResult(BaseModel):
    run_id: str | None = None
    run_name: str | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None
    items: list[HostedRunItemResult] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


def _normalize_items(raw_items: Any) -> list[HostedRunItemResult]:
    normalized: list[HostedRunItemResult] = []
    if not isinstance(raw_items, list):
        return normalized
    for entry in raw_items:
        if isinstance(entry, Mapping):
            item = cast(Mapping[str, object], entry.get("item") or entry.get("input") or {})
            output = cast(Mapping[str, object] | dict[str, object] | None, entry.get("output") or entry.get("result"))
            trace_id = cast(str | None, entry.get("trace_id") or entry.get("traceId") or entry.get("id"))
            normalized.append(HostedRunItemResult(item=item, output=output, trace_id=trace_id))
        else:
            try:
                item_val = cast(Mapping[str, object], getattr(entry, "item", {}) or getattr(entry, "input", {}) or {})
                output_val = cast(Mapping[str, object] | dict[str, object] | None, getattr(entry, "output", None) or getattr(entry, "result", None))
                trace_val = cast(str | None, getattr(entry, "trace_id", None) or getattr(entry, "traceId", None) or getattr(entry, "id", None))
                normalized.append(HostedRunItemResult(item=item_val, output=output_val, trace_id=trace_val))
            except Exception:
                continue
    return normalized


def run_hosted_dataset_experiment(
    dataset_name: str,
    dataset_version: str | None,
    experiment_name: str,
    task_fn: Callable[[Mapping[str, object], object | None], Mapping[str, object] | dict[str, object]],
    evaluators: list[object] | None = None,
    run_evaluators: list[object] | None = None,
    metadata: Mapping[str, object] | None = None,
    max_concurrency: int | None = None,
) -> HostedRunResult:
    client = get_langfuse_client()
    run_name = f"{experiment_name}-{int(time.time())}"
    client_any = cast(Any, client)
    dataset = client_any.get_dataset(name=dataset_name, version=dataset_version)
    run_kwargs: dict[str, Any] = {
        "name": experiment_name,
        "run_name": run_name,
        "task": task_fn,
        "metadata": dict(metadata) if metadata else None,
    }
    if evaluators is not None:
        run_kwargs["evaluators"] = evaluators
    if run_evaluators is not None:
        run_kwargs["run_evaluators"] = run_evaluators
    if max_concurrency is not None:
        run_kwargs["max_concurrency"] = max_concurrency
    result = cast(Any, dataset).run_experiment(**run_kwargs)
    run_id = cast(str | None, getattr(result, "id", None) or getattr(result, "run_id", None))
    items_raw = getattr(result, "items", None) or getattr(result, "item_results", None)
    hosted_result = HostedRunResult(
        run_id=run_id,
        run_name=run_name,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        items=_normalize_items(items_raw),
        metadata={"run_kind": experiment_name, **(dict(metadata) if metadata else {})},
    )
    flush_langfuse()
    return hosted_result


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
    name: str | HostedJCBaselineRequest, version: str | None = None
) -> BaselineBundle:
    if isinstance(name, HostedJCBaselineRequest):
        req = name
    else:
        session_cfg = SessionConfig(
            session_scope="item",
            allow_generated_fallback=True,
        )
        req = HostedJCBaselineRequest(dataset=name, version=version, session=session_cfg)

    def _run_item(
        item: Mapping[str, object], parent_trace: object | None = None
    ) -> Mapping[str, object]:
        dataset_item = normalize_dataset_item(item)
        with propagate_context(
            trace_name="jc_baseline_item",
            session_id=resolve_session_id(req.session, run_id=req.dataset, item_id=dataset_item.id),
            metadata={"dataset": req.dataset, "dataset_version": req.version},
        ):
            snapshot = compute_baseline_for_item(
                dataset_item,
                dataset_name=req.dataset,
                dataset_version=req.version,
                parent_trace=parent_trace,
            )
            return {"baseline": snapshot.model_dump()}

    hosted = run_hosted_dataset_experiment(
        dataset_name=req.dataset,
        dataset_version=req.version,
        experiment_name="jc_baseline",
        task_fn=_run_item,
        metadata={"run_kind": "jc_baseline", "session_id": resolve_session_id(req.session, run_id=req.dataset)},
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
    session_config: SessionConfig | None,
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
        resolved_session = resolve_session_id(
            session_config,
            run_id=dataset_name or "",
            item_id=normalized_item.id or normalized_item.repo or normalized_item.symbol,
        )
        with propagate_context(
            trace_name="ic_optimization_item",
            session_id=resolved_session,
            metadata={
                "dataset": dataset_name,
                "dataset_version": dataset_version,
                "item_id": normalized_item.id,
                "repo": normalized_item.repo,
                "symbol": normalized_item.symbol,
            },
        ):
            history = run_loop(
                task,
                iterations=iterations,
                parent_trace=parent_trace,
                baseline_snapshot=baseline,
                session_id=resolved_session,
            )
        metrics: Mapping[str, object] = {}
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
    req: str | HostedICOptimizationRequest,
    version: str | None = None,
    iterations: int = 5,
    baselines: BaselineBundle | None = None,
    recompute_baselines: bool = False,
) -> list[dict[str, object]]:
    if isinstance(req, HostedICOptimizationRequest):
        request = req
    else:
        session_cfg = SessionConfig(session_scope="item", allow_generated_fallback=True)
        request = HostedICOptimizationRequest(
            dataset=req,
            version=version,
            iterations=iterations,
            baselines=baselines,
            recompute_baselines=recompute_baselines,
            session=session_cfg,
        )
    items = fetch_dataset_items(name=request.dataset, version=request.version)
    if request.baselines is None and not request.recompute_baselines:
        raise ValueError(
            "Baselines bundle required for optimization; set recompute_baselines=True to compute."
        )
    bundle = (
        request.baselines
        if request.baselines is not None
        else run_hosted_jc_baseline_experiment(
            HostedJCBaselineRequest(
                dataset=request.dataset,
                version=request.version,
                session=request.session
                or SessionConfig(session_scope="item", allow_generated_fallback=True),
            )
        )
    )
    return _run_ic_optimizations(
        items,
        bundle,
        dataset_name=request.dataset,
        dataset_version=request.version,
        iterations=request.iterations,
        hosted=True,
        session_config=request.session,
    )
