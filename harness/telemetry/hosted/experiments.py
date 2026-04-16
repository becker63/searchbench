from __future__ import annotations

import concurrent.futures
from concurrent.futures import FIRST_COMPLETED
from typing import Mapping, Sequence

from pydantic import BaseModel, Field

from harness.localization.runtime.evaluate import (
    LocalizationEvaluationFailure,
    evaluate_localization_batch,
)
from harness.localization.models import LCATask, normalize_lca_task
from harness.localization.scoring_models import TaskScoreSummary, summarize_task_score
from harness.localization.token_usage import TokenUsageRecord
from harness.localization.materialization.materialize import RepoMaterializationResult, RepoMaterializer
from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory
from harness.telemetry.hosted.baselines import (
    BaselineBundle,
    BaselineSnapshot,
    compute_baseline_for_task,
    emit_baseline_dataset_aggregates,
    load_persisted_baseline_bundle,
    make_baseline_bundle,
    persist_baseline_bundle,
)
from harness.telemetry.hosted.hf_lca import fetch_hf_localization_dataset
from harness.telemetry.tracing import flush_langfuse, propagate_context, start_observation
from harness.telemetry.tracing.session_policy import resolve_session_id
from harness.telemetry.policy_reducer import PolicyReducerSummary, build_task_input, reduce_global
from harness.utils.env import get_runner_model

import logging

_LOGGER = logging.getLogger(__name__)

from harness.entrypoints.models.requests import HostedLocalizationBaselineRequest, HostedLocalizationRunRequest


class LocalizationRunItemResult(BaseModel):
    task: Mapping[str, object]
    score_summary: TaskScoreSummary
    prediction: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    materialization: RepoMaterializationResult | None = None
    token_usage: TokenUsageRecord | None = None


class LocalizationRunResult(BaseModel):
    dataset_name: str
    dataset_version: str | None = None
    items: list[LocalizationRunItemResult] = Field(default_factory=list)
    failure: LocalizationEvaluationFailure | None = None
    reducer_summary: PolicyReducerSummary | None = None


def _resolve_items(req: HostedLocalizationRunRequest) -> list[LCATask]:
    return fetch_hf_localization_dataset(
        req.dataset,
        dataset_config=req.dataset_config,
        dataset_split=req.dataset_split,
        revision=req.version,
    )


def select_localization_tasks(
    tasks: Sequence[LCATask],
    max_items: int | None = None,
    offset: int = 0,
) -> tuple[list[LCATask], dict[str, object]]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be > 0 when provided")
    sorted_tasks = sorted(list(tasks), key=lambda t: getattr(t, "task_id", None) or t.identity.task_id())
    total_available = len(sorted_tasks)
    start = min(offset, total_available)
    end = total_available if max_items is None else min(start + max_items, total_available)
    selected = sorted_tasks[start:end]
    preview = [getattr(t, "task_id", None) or t.identity.task_id() for t in selected[:5]]
    selection_info: dict[str, object] = {
        "total_available": total_available,
        "offset": offset,
        "limit": max_items,
        "selected_count": len(selected),
        "preview": preview,
    }
    return selected, selection_info


class _TaskFailure(Exception):
    def __init__(self, index: int, task_id: str | None, category: LocalizationFailureCategory | str, message: str):
        super().__init__(message)
        self.index = index
        self.task_id = task_id
        self.category = category
        self.message = message


def _effective_workers(requested: int, selected_count: int) -> int:
    if requested is None or requested <= 0:
        raise ValueError("max_workers must be a positive integer")
    if selected_count <= 0:
        raise ValueError("No tasks selected for execution")
    return max(1, min(requested, selected_count))


def _normalize_failure_category(category: LocalizationFailureCategory | str) -> LocalizationFailureCategory:
    if isinstance(category, LocalizationFailureCategory):
        return category
    try:
        return LocalizationFailureCategory(category)
    except Exception:
        return LocalizationFailureCategory.UNKNOWN


def run_hosted_localization_baseline(
    req: HostedLocalizationBaselineRequest,
    materializer: RepoMaterializer | None = None,
    tasks: list[LCATask] | None = None,
) -> BaselineBundle:
    run_session_id = resolve_session_id(req.session, run_id=req.dataset)
    resolved_model = get_runner_model()
    _LOGGER.info("Localization baseline runner model resolved to %s", resolved_model)
    cached_bundle = load_persisted_baseline_bundle(
        dataset_name=req.dataset,
        dataset_config=req.dataset_config,
        dataset_split=req.dataset_split,
        dataset_version=req.version,
    )
    selected_tasks: list[LCATask] = []
    snapshots: list[BaselineSnapshot] = []
    failure: LocalizationEvaluationFailure | None = None
    effective_workers = 1
    selection_info: Mapping[str, object] | None = None
    if cached_bundle is None:
        tasks = tasks or fetch_hf_localization_dataset(
            req.dataset,
            dataset_config=req.dataset_config,
            dataset_split=req.dataset_split,
            revision=req.version,
        )
        if not tasks:
            raise RuntimeError("No localization tasks available for baseline run")
        if req.selection:
            selected_tasks = list(tasks)
            selection_info = req.selection
        else:
            selected_tasks, selection_info = select_localization_tasks(tasks, max_items=req.max_items, offset=req.offset)
        effective_workers = _effective_workers(req.max_workers or 1, len(selected_tasks))
        selection_meta = selection_info or {
            "selected_count": len(selected_tasks),
            "offset": req.offset,
            "limit": req.max_items,
        }
    else:
        snapshots = list(cached_bundle.items)
        failure = cached_bundle.failure
        selection_meta = {"selected_count": len(snapshots)}
    with propagate_context(
        trace_name="localization_baseline_dataset",
        session_id=run_session_id,
        metadata={
            "dataset": req.dataset,
            "dataset_version": req.version,
            "dataset_config": req.dataset_config,
            "dataset_split": req.dataset_split,
            "run_kind": "localization_baseline",
            "dataset_source": "huggingface",
        },
    ):
        with start_observation(
            name="localization_baseline_dataset",
            metadata={
                "dataset": req.dataset,
                "dataset_version": req.version,
                "dataset_config": req.dataset_config,
                "dataset_split": req.dataset_split,
                "run_kind": "localization_baseline",
                "dataset_source": "huggingface",
                "selection": selection_meta,
                "runner_model": resolved_model,
            },
        ) as dataset_span:
            parent_trace = dataset_span
            if cached_bundle is None:
                if effective_workers == 1:
                    for task in selected_tasks:
                        snapshots.append(
                            compute_baseline_for_task(
                                task,
                                session_id=run_session_id,
                                dataset_version=req.version,
                                parent_trace=parent_trace,
                                dataset_source="huggingface",
                                materializer=materializer,
                            )
                        )
                else:
                    snapshots, failure = _run_baseline_parallel(
                        selected_tasks,
                        effective_workers,
                        parent_trace,
                        materializer,
                        req.version,
                        run_session_id,
                    )
            emit_baseline_dataset_aggregates(
                dataset_span,
                snapshots,
                extra_metadata={
                    "cache_hit": cached_bundle is not None,
                    "dataset": req.dataset,
                    "dataset_config": req.dataset_config,
                    "dataset_split": req.dataset_split,
                    "dataset_version": req.version,
                    "selection": selection_meta,
                    "runner_model": resolved_model,
                },
            )
    flush_langfuse()
    if cached_bundle is not None:
        return cached_bundle
    bundle = make_baseline_bundle(
        snapshots,
        dataset_name=req.dataset,
        dataset_version=req.version,
        metadata={"run_kind": "localization_baseline"},
        failure=failure,
    )
    persist_baseline_bundle(
        bundle,
        dataset_name=req.dataset,
        dataset_config=req.dataset_config,
        dataset_split=req.dataset_split,
        dataset_version=req.version,
    )
    return bundle


def _baseline_worker(
    index: int,
    task: LCATask,
    parent_trace: object,
    materializer: RepoMaterializer | None,
    dataset_version: str | None,
    session_id: str | None,
) -> BaselineSnapshot:
    if parent_trace is None:
        raise _TaskFailure(index, getattr(task, "task_id", None), LocalizationFailureCategory.UNKNOWN, "missing parent trace")
    try:
        return compute_baseline_for_task(
            task,
            session_id=session_id,
            dataset_version=dataset_version,
            parent_trace=parent_trace,
            dataset_source="huggingface",
            materializer=materializer,
        )
    except LocalizationEvaluationError as exc:
        category = getattr(exc, "category", LocalizationFailureCategory.UNKNOWN)
        raise _TaskFailure(index, getattr(task, "task_id", None), category, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _TaskFailure(index, getattr(task, "task_id", None), LocalizationFailureCategory.UNKNOWN, str(exc)) from exc


def _run_baseline_parallel(
    tasks: list[LCATask],
    effective_workers: int,
    parent_trace: object,
    materializer: RepoMaterializer | None,
    dataset_version: str | None,
    session_id: str | None,
) -> tuple[list[BaselineSnapshot], LocalizationEvaluationFailure | None]:
    results: list[BaselineSnapshot | None] = [None] * len(tasks)
    failure: _TaskFailure | None = None
    next_index = 0
    inflight: dict[concurrent.futures.Future[BaselineSnapshot], int] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        def _submit(idx: int) -> None:
            future = executor.submit(
                _baseline_worker,
                idx,
                tasks[idx],
                parent_trace,
                materializer,
                dataset_version,
                session_id,
            )
            inflight[future] = idx

        while next_index < len(tasks) and len(inflight) < effective_workers:
            _submit(next_index)
            next_index += 1

        while inflight:
            done, _ = concurrent.futures.wait(inflight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                idx = inflight.pop(future)
                try:
                    results[idx] = future.result()
                except _TaskFailure as exc:
                    if failure is None or exc.index < failure.index:
                        failure = exc
                except Exception as exc:  # noqa: BLE001
                    fallback = _TaskFailure(idx, getattr(tasks[idx], "task_id", None), LocalizationFailureCategory.UNKNOWN, str(exc))
                    if failure is None or idx < failure.index:
                        failure = fallback
                if failure is None and next_index < len(tasks):
                    _submit(next_index)
                    next_index += 1

    failure_payload: LocalizationEvaluationFailure | None = None
    if failure is not None:
        failure_payload = LocalizationEvaluationFailure(
            category=_normalize_failure_category(failure.category),
            message=failure.message,
            task_id=failure.task_id,
        )
    ordered_results = [r for r in results if r is not None]
    return ordered_results, failure_payload


def _normalize_task_payload(task: LCATask | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(task, Mapping):
        return task
    if hasattr(task, "model_dump"):
        return task.model_dump()
    raise RuntimeError("Localization task must be a mapping or Pydantic model")


def _run_experiment_task(
    task: LCATask | Mapping[str, object],
    parent_trace: object,
    materializer: RepoMaterializer | None,
    session_id: str | None,
) -> LocalizationRunItemResult:
    if parent_trace is None:
        raise _TaskFailure(-1, getattr(task, "task_id", None), LocalizationFailureCategory.UNKNOWN, "missing parent trace")
    task_payload = _normalize_task_payload(task)
    normalized_task = normalize_lca_task(task_payload)
    try:
        with propagate_context(session_id=session_id):
            eval_result = evaluate_localization_batch(
                tasks=[normalized_task],
                dataset_source="huggingface",
                materializer=materializer,
                parent_trace=parent_trace,
            )
        if eval_result.failure or not eval_result.items:
            failure = eval_result.failure or LocalizationEvaluationFailure(
                category=LocalizationFailureCategory.UNKNOWN, message="experiment_evaluation_failed", task_id=normalized_task.task_id
            )
            raise _TaskFailure(-1, failure.task_id, failure.category, failure.message)
        task_result = eval_result.items[0]
        score_summary = summarize_task_score(normalized_task.task_id, task_result.score_bundle)
        return LocalizationRunItemResult(
            task=normalized_task.model_dump(),
            score_summary=score_summary,
            prediction=task_result.prediction,
            trace_id=getattr(parent_trace, "id", None),
            materialization=task_result.materialization,
            token_usage=task_result.token_usage,
        )
    except _TaskFailure:
        raise
    except LocalizationEvaluationError as exc:
        category = getattr(exc, "category", LocalizationFailureCategory.UNKNOWN)
        raise _TaskFailure(-1, getattr(normalized_task, "task_id", None), category, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _TaskFailure(-1, getattr(normalized_task, "task_id", None), LocalizationFailureCategory.UNKNOWN, str(exc)) from exc


def _run_experiment_parallel(
    tasks: list[LCATask],
    effective_workers: int,
    parent_trace: object,
    materializer: RepoMaterializer | None,
    session_id: str | None,
) -> tuple[list[LocalizationRunItemResult], LocalizationEvaluationFailure | None]:
    results: list[LocalizationRunItemResult | None] = [None] * len(tasks)
    failure: _TaskFailure | None = None
    next_index = 0
    inflight: dict[concurrent.futures.Future[LocalizationRunItemResult], int] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        def _submit(idx: int) -> None:
            future = executor.submit(
                _run_experiment_task,
                tasks[idx],
                parent_trace,
                materializer,
                session_id,
            )
            inflight[future] = idx

        while next_index < len(tasks) and len(inflight) < effective_workers:
            _submit(next_index)
            next_index += 1

        while inflight:
            done, _ = concurrent.futures.wait(inflight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                idx = inflight.pop(future)
                try:
                    results[idx] = future.result()
                except _TaskFailure as exc:
                    indexed_failure = _TaskFailure(idx, exc.task_id, exc.category, exc.message)
                    if failure is None or indexed_failure.index < failure.index:
                        failure = indexed_failure
                except Exception as exc:  # noqa: BLE001
                    fallback = _TaskFailure(idx, getattr(tasks[idx], "task_id", None), LocalizationFailureCategory.UNKNOWN, str(exc))
                    if failure is None or idx < failure.index:
                        failure = fallback
                if failure is None and next_index < len(tasks):
                    _submit(next_index)
                    next_index += 1

    failure_payload: LocalizationEvaluationFailure | None = None
    if failure is not None:
        failure_payload = LocalizationEvaluationFailure(
            category=_normalize_failure_category(failure.category),
            message=failure.message,
            task_id=failure.task_id,
        )
    ordered_results = [r for r in results if r is not None]
    return ordered_results, failure_payload

def run_hosted_localization_experiment(
    req: HostedLocalizationRunRequest,
    materializer: RepoMaterializer | None = None,
    tasks: list[LCATask] | None = None,
) -> LocalizationRunResult:
    tasks = tasks or _resolve_items(req)
    if not tasks:
        raise RuntimeError("No localization tasks available for experiment run")
    if req.selection:
        selected_tasks = list(tasks)
        selection_info = req.selection
    else:
        selected_tasks, selection_info = select_localization_tasks(tasks, max_items=req.max_items, offset=req.offset)
    results: list[LocalizationRunItemResult] = []
    failure: LocalizationEvaluationFailure | None = None
    effective_workers = _effective_workers(req.max_workers or 1, len(selected_tasks))
    selection_meta = selection_info or {
        "selected_count": len(selected_tasks),
        "offset": req.offset,
        "limit": req.max_items,
    }
    projection_meta: dict[str, object] = {"provided": bool(req.projection)}
    run_session_id = resolve_session_id(req.session, run_id=req.dataset)
    with propagate_context(
        trace_name="localization_experiment",
        session_id=run_session_id,
        metadata={
            "dataset": req.dataset,
            "dataset_version": req.version,
            "dataset_config": req.dataset_config,
            "dataset_split": req.dataset_split,
            "run_kind": "localization_experiment",
            "dataset_source": "huggingface",
        },
    ):
        with start_observation(
            name="localization_experiment_dataset",
            metadata={
                "dataset": req.dataset,
                "dataset_version": req.version,
                "dataset_config": req.dataset_config,
                "dataset_split": req.dataset_split,
                "run_kind": "localization_experiment",
                "dataset_source": "huggingface",
                "selection": selection_meta,
                "projection": projection_meta,
            },
        ) as dataset_span:
            parent_trace = dataset_span
            if effective_workers == 1:
                for task in selected_tasks:
                    results.append(
                        _run_experiment_task(
                            task=task,
                            parent_trace=parent_trace,
                            materializer=materializer,
                            session_id=run_session_id,
                        )
                    )
            else:
                results, failure = _run_experiment_parallel(
                    selected_tasks,
                    effective_workers,
                    parent_trace,
                    materializer,
                    run_session_id,
                )
    flush_langfuse()
    reducer_summary = _build_reducer_summary(results)
    return LocalizationRunResult(dataset_name=req.dataset, dataset_version=req.version, items=results, failure=failure, reducer_summary=reducer_summary)


def _build_reducer_summary(results: list[LocalizationRunItemResult]) -> PolicyReducerSummary | None:
    if not results:
        return None
    task_inputs = []
    for item in results:
        try:
            normalized_task = normalize_lca_task(item.task)
        except Exception:
            continue
        task_inputs.append(
            build_task_input(
                identity=normalized_task.identity,
                task_id=normalized_task.task_id,
                score_summary=item.score_summary,
                candidate_usage=item.token_usage,
                baseline_usage=None,
            )
        )
    if not task_inputs:
        return None
    return reduce_global(task_inputs)
