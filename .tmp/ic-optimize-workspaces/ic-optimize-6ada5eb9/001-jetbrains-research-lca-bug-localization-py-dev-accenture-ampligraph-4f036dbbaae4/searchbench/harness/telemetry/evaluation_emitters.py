from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from langfuse import LangfuseGeneration, LangfuseSpan

from harness.telemetry.observability_models import (
    EvaluationTelemetryArtifact,
    OptimizeIterationTelemetryContext,
    evaluation_metrics_map,
)
from harness.telemetry.tracing.score_emitter import emit_score_for_handle

if TYPE_CHECKING:
    from harness.orchestration.types import EvaluationMetrics

_LOGGER = logging.getLogger(__name__)


def _observation_name(handle: LangfuseSpan | LangfuseGeneration | None) -> str:
    if handle is None:
        return "none"
    return handle.id


def safe_update_observation_metadata(
    handle: LangfuseSpan | LangfuseGeneration | None,
    metadata: Mapping[str, object],
    *,
    operation: str,
    observation_name: str | None = None,
) -> None:
    if handle is None:
        return
    keys = sorted(str(key) for key in metadata)
    try:
        handle.update(metadata=dict(metadata))
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "Langfuse metadata update failed operation=%s observation=%s keys=%s error=%s",
            operation,
            observation_name or _observation_name(handle),
            keys,
            exc,
        )


def safe_emit_score_for_handle(
    handle: LangfuseSpan | LangfuseGeneration | None,
    *,
    name: str,
    value: float | int | bool,
    data_type: str | None = None,
    comment: str | None = None,
    operation: str,
) -> None:
    try:
        emit_score_for_handle(
            handle,
            name=name,
            value=value,
            data_type=data_type,
            comment=comment,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "Langfuse score emission failed operation=%s observation=%s score=%s error=%s",
            operation,
            _observation_name(handle),
            name,
            exc,
        )


def _score_type(value: float | bool) -> str:
    return "BOOLEAN" if isinstance(value, bool) else "NUMERIC"


def _score_value(value: float | bool) -> float | int:
    if isinstance(value, bool):
        return 1 if value else 0
    return float(value)


def emit_localization_task_telemetry(
    handle: LangfuseSpan | LangfuseGeneration | None,
    artifact: EvaluationTelemetryArtifact,
    *,
    score_prefix: str,
    telemetry_payload: Mapping[str, object],
    predicted_files: list[str],
    changed_files: list[str],
    task_summary: Mapping[str, object] | None = None,
) -> None:
    """Emit localization evaluation on the `localization_task` observation.

    Langfuse scores are the queryable evaluation objects. Metadata is kept
    observation-local and compact so backend spans remain diagnostic-only.
    """

    for metric_name, value in artifact.metric_map.items():
        safe_emit_score_for_handle(
            handle,
            name=f"{score_prefix}.{metric_name}",
            value=_score_value(value),
            data_type=_score_type(value),
            operation="localization_task.score",
        )

    metadata: dict[str, object] = {
        "telemetry": dict(telemetry_payload),
        "score_summary": artifact.score_summary.model_dump(mode="json"),
        "score_context": artifact.compact_score_context.model_dump(mode="json", exclude_none=True),
        "predicted_files": list(predicted_files),
        "changed_files": list(changed_files),
        "score_metric_names": sorted(artifact.metric_map),
    }
    if task_summary is not None:
        metadata["task_summary"] = dict(task_summary)
    safe_update_observation_metadata(
        handle,
        metadata,
        operation="localization_task.metadata",
        observation_name="localization_task",
    )


def emit_optimize_iteration_telemetry(
    handle: LangfuseSpan | LangfuseGeneration | None,
    *,
    metrics: EvaluationMetrics,
    iteration: int,
    status: str | None = None,
    pipeline_passed: bool | None = None,
    score_context_summary: Mapping[str, object] | None = None,
    error: str | None = None,
) -> None:
    metric_map = evaluation_metrics_map(metrics)
    if pipeline_passed is not None:
        metric_map["pipeline_passed"] = bool(pipeline_passed)
    comment = json.dumps({"iteration": iteration})
    for metric_name, value in metric_map.items():
        safe_emit_score_for_handle(
            handle,
            name=metric_name,
            value=_score_value(value),
            data_type=_score_type(value),
            comment=comment,
            operation="optimize_iteration.score",
        )
    context = OptimizeIterationTelemetryContext(
        iteration=iteration,
        iteration_ordinal=iteration + 1,
        status=status,
        pipeline_passed=pipeline_passed,
        score_context_summary=None,
        error=error,
    ).model_dump(mode="json", exclude_none=True)
    metadata: dict[str, object] = {
        "iteration": iteration,
        "iteration_ordinal": iteration + 1,
        "status": status,
        "pipeline_passed": pipeline_passed,
        "evaluation_metric_names": sorted(metric_map),
        "evaluation": context,
    }
    if score_context_summary:
        metadata["score_context_summary"] = dict(score_context_summary)
    if error:
        metadata["error"] = error
    safe_update_observation_metadata(
        handle,
        {key: value for key, value in metadata.items() if value is not None},
        operation="optimize_iteration.metadata",
        observation_name="optimize_iteration",
    )


__all__ = [
    "emit_localization_task_telemetry",
    "emit_optimize_iteration_telemetry",
    "safe_emit_score_for_handle",
    "safe_update_observation_metadata",
]
