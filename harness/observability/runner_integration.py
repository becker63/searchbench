from __future__ import annotations

import time
from typing import Any, Callable, Mapping, cast

from pydantic import BaseModel, Field

from .langfuse import get_langfuse_client, start_span, start_trace


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
    if client:
        try:
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
            return HostedRunResult(
                run_id=run_id,
                run_name=run_name,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                items=_normalize_items(items_raw),
                metadata={"run_kind": experiment_name, **(dict(metadata) if metadata else {})},
            )
        except Exception:
            pass

    # Defensive fallback: manual trace-run if client unavailable/error.
    root = start_trace(
        f"{experiment_name}_hosted_fallback",
        metadata={"dataset": dataset_name, "dataset_version": dataset_version, "run_kind": experiment_name, "count": len([])},
    )
    run_id = getattr(root, "id", None)
    items: list[HostedRunItemResult] = []
    return HostedRunResult(
        run_id=run_id,
        run_name=run_name,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        items=items,
        metadata={"run_kind": experiment_name, "fallback": True, **(dict(metadata) if metadata else {})},
    )


def run_local_dataset_experiment(
    data: list[Mapping[str, object]],
    experiment_name: str,
    task_fn: Callable[[Mapping[str, object], object | None], Mapping[str, object] | dict[str, object]],
    evaluators: list[object] | None = None,
    run_evaluators: list[object] | None = None,
    metadata: Mapping[str, object] | None = None,
    max_concurrency: int | None = None,
) -> HostedRunResult:
    client = get_langfuse_client()
    run_name = f"{experiment_name}-local-{int(time.time())}"
    if client:
        try:
            client_any = cast(Any, client)
            run_kwargs: dict[str, Any] = {
                "name": experiment_name,
                "data": data,
                "task": task_fn,
                "metadata": dict(metadata) if metadata else None,
            }
            if evaluators is not None:
                run_kwargs["evaluators"] = evaluators
            if run_evaluators is not None:
                run_kwargs["run_evaluators"] = run_evaluators
            if max_concurrency is not None:
                run_kwargs["max_concurrency"] = max_concurrency
            result = client_any.run_experiment(**run_kwargs)
            run_id = cast(str | None, getattr(result, "id", None) or getattr(result, "run_id", None))
            items_raw = getattr(result, "items", None) or getattr(result, "item_results", None)
            return HostedRunResult(
                run_id=run_id,
                run_name=run_name,
                dataset_name=None,
                dataset_version=None,
                items=_normalize_items(items_raw),
                metadata={"run_kind": experiment_name, **(dict(metadata) if metadata else {})},
            )
        except Exception:
            pass

    # Defensive local fallback: run tasks with Langfuse spans so traces still exist.
    root = start_trace(
        f"{experiment_name}_local_fallback",
        metadata={"run_kind": experiment_name, "count": len(data)},
    )
    items: list[HostedRunItemResult] = []
    for entry in data:
        span = start_span(root, f"{experiment_name}_item", metadata={"run_kind": experiment_name, **(dict(metadata) if metadata else {})})
        output = task_fn(entry, span)
        items.append(HostedRunItemResult(item=entry, output=output, trace_id=getattr(span, "id", None)))
    return HostedRunResult(
        run_id=getattr(root, "id", None),
        run_name=run_name,
        dataset_name=None,
        dataset_version=None,
        items=items,
        metadata={"run_kind": experiment_name, "fallback": True, **(dict(metadata) if metadata else {})},
    )


__all__ = ["HostedRunResult", "run_hosted_dataset_experiment", "run_local_dataset_experiment"]
