"""Telemetry: tracing, scoring, datasets, and baselines."""

from importlib import import_module
from typing import Any, TYPE_CHECKING

__all__ = [
    "flush_langfuse",
    "get_langfuse_client",
    "get_tracing_openai_client",
    "is_langfuse_enabled",
    "propagate_context",
    "start_observation",
    "start_root_observation",
    "start_child_observation",
    "emit_score",
    "emit_score_for_handle",
    "ScorePayload",
    "LocalizationDataset",
    "LocalizationDatasetLoadError",
    "fetch_localization_dataset",
    "langfuse_dataset_item_from_task",
    "local_dataset",
    "normalize_localization_task",
    "task_from_langfuse_dataset_item",
    "EvaluationBundle",
    "EvaluationRecord",
    "compute_baseline_for_task",
    "make_baseline_bundle",
    "require_baseline",
    "resolve_baseline",
]

if TYPE_CHECKING:
    from harness.telemetry.tracing import (
        flush_langfuse,
        get_langfuse_client,
        get_tracing_openai_client,
        is_langfuse_enabled,
        propagate_context,
        start_child_observation,
        start_observation,
        start_root_observation,
    )
    from harness.telemetry.tracing.score_emitter import (
        ScorePayload,
        emit_score,
        emit_score_for_handle,
    )
    from harness.telemetry.hosted.datasets import (
        LocalizationDataset,
        LocalizationDatasetLoadError,
        fetch_localization_dataset,
        langfuse_dataset_item_from_task,
        local_dataset,
        normalize_localization_task,
        task_from_langfuse_dataset_item,
    )
    from harness.telemetry.hosted.baselines import (
        EvaluationBundle,
        EvaluationRecord,
        compute_baseline_for_task,
        make_baseline_bundle,
        require_baseline,
        resolve_baseline,
    )


_TRACING_EXPORTS = {
    "flush_langfuse",
    "get_langfuse_client",
    "get_tracing_openai_client",
    "is_langfuse_enabled",
    "propagate_context",
    "start_observation",
    "start_root_observation",
    "start_child_observation",
}

_SCORE_EXPORTS = {
    "emit_score",
    "emit_score_for_handle",
    "ScorePayload",
}

_BASELINE_EXPORTS = {
    "EvaluationBundle",
    "EvaluationRecord",
    "compute_baseline_for_task",
    "make_baseline_bundle",
    "require_baseline",
    "resolve_baseline",
}

_DATASET_EXPORTS = {
    "LocalizationDataset",
    "LocalizationDatasetLoadError",
    "fetch_localization_dataset",
    "langfuse_dataset_item_from_task",
    "local_dataset",
    "normalize_localization_task",
    "task_from_langfuse_dataset_item",
}


def __getattr__(name: str) -> Any:
    if name in _TRACING_EXPORTS:
        module = import_module("harness.telemetry.tracing")
        return getattr(module, name)
    if name in _SCORE_EXPORTS:
        module = import_module("harness.telemetry.tracing.score_emitter")
        return getattr(module, name)
    if name in _BASELINE_EXPORTS:
        module = import_module("harness.telemetry.hosted.baselines")
        return getattr(module, name)
    if name in _DATASET_EXPORTS:
        module = import_module("harness.telemetry.hosted.datasets")
        return getattr(module, name)
    raise AttributeError(f"module 'harness.telemetry' has no attribute {name!r}")
