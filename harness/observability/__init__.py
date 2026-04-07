from importlib import import_module
from typing import Any

from .langfuse import (  # noqa: F401
    flush_langfuse,
    get_langfuse_client,
    get_tracing_openai_client,
    is_langfuse_enabled,
    propagate_context,
    start_child_observation,
    start_observation,
    start_root_observation,
)
from .score_emitter import ScorePayload, emit_score, emit_score_for_handle  # noqa: F401
from .datasets import LocalizationDataset, fetch_localization_dataset, local_dataset, normalize_localization_task  # noqa: F401

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
    # Baseline exports are loaded lazily to avoid circular imports during package init.
    "BaselineBundle",
    "BaselineSnapshot",
    "baseline_key",
    "compute_baseline_for_task",
    "make_baseline_bundle",
    "require_baseline",
    "resolve_baseline",
]


def __getattr__(name: str) -> Any:
    """
    Lazily import baseline helpers to avoid circular imports when submodules depend on localization.
    """
    if name in {
        "BaselineBundle",
        "BaselineSnapshot",
        "baseline_key",
        "compute_baseline_for_task",
        "make_baseline_bundle",
        "require_baseline",
        "resolve_baseline",
    }:
        module = import_module("harness.observability.baselines")
        return getattr(module, name)
    raise AttributeError(f"module 'harness.observability' has no attribute {name!r}")
