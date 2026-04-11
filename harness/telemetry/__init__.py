"""Telemetry: tracing, scoring, datasets, and baselines."""

from importlib import import_module
from typing import Any, TYPE_CHECKING

from .tracing import (
    flush_langfuse,
    get_langfuse_client,
    get_tracing_openai_client,
    is_langfuse_enabled,
    propagate_context,
    start_child_observation,
    start_observation,
    start_root_observation,
)
from .tracing.score_emitter import ScorePayload, emit_score, emit_score_for_handle
from .hosted.datasets import (
    LocalizationDataset,
    fetch_localization_dataset,
    local_dataset,
    normalize_localization_task,
)

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
    "fetch_localization_dataset",
    "local_dataset",
    "normalize_localization_task",
    "BaselineBundle",
    "BaselineSnapshot",
    "baseline_key",
    "compute_baseline_for_task",
    "make_baseline_bundle",
    "require_baseline",
    "resolve_baseline",
]

if TYPE_CHECKING:
    from harness.telemetry.hosted.baselines import (
        BaselineBundle,
        BaselineSnapshot,
        baseline_key,
        compute_baseline_for_task,
        make_baseline_bundle,
        require_baseline,
        resolve_baseline,
    )


def __getattr__(name: str) -> Any:
    if name in {
        "BaselineBundle",
        "BaselineSnapshot",
        "baseline_key",
        "compute_baseline_for_task",
        "make_baseline_bundle",
        "require_baseline",
        "resolve_baseline",
    }:
        module = import_module("harness.telemetry.hosted.baselines")
        return getattr(module, name)
    raise AttributeError(f"module 'harness.telemetry' has no attribute {name!r}")
