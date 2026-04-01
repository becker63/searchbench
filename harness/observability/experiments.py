from __future__ import annotations

"""
Application/experiment boundary: hosted JC baseline and IC optimization wrappers.
Accept normalized inputs or typed requests; do not embed CLI or leaf logic.
"""

from typing import Mapping, Sequence

from ..entrypoints import HostedICOptimizationRequest, HostedJCBaselineRequest
from ..loop import iteration_record_to_public_dict, run_loop
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
from .langfuse import flush_langfuse, propagate_context
from .runner_integration import HostedRunResult, run_hosted_dataset_experiment
from .score_emitter import emit_score_for_handle
from .session_policy import SessionConfig, resolve_session_id


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
