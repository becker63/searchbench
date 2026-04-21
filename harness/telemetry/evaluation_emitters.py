from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from langfuse import LangfuseGeneration, LangfuseSpan
from langfuse.api.commons.types import ScoreDataType

from harness.log import get_logger
from harness.telemetry.observability_models import (
    EvaluationTelemetryArtifact,
    LOCALIZATION_CANONICAL_COMPONENTS,
    OptimizeIterationTelemetryContext,
    ScoreComponentTelemetryState,
    optimize_component_states,
)

if TYPE_CHECKING:
    from harness.orchestration.types import EvaluationMetrics

_LOGGER = get_logger(__name__)
ObservationHandle = LangfuseSpan | LangfuseGeneration


def _observation_name(handle: ObservationHandle | None) -> str:
    if handle is None:
        return "none"
    return handle.id


def _update_observation_metadata(
    handle: ObservationHandle | None,
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


def _emit_score_for_handle(
    handle: ObservationHandle | None,
    *,
    name: str,
    value: float | int | bool,
    data_type: Literal[ScoreDataType.NUMERIC, ScoreDataType.BOOLEAN],
    comment: str | None = None,
    operation: str,
) -> None:
    if handle is None:
        return
    try:
        if data_type == ScoreDataType.BOOLEAN:
            handle.score(
                name=name,
                value=bool(value),
                data_type=ScoreDataType.BOOLEAN,
                comment=comment,
            )
        else:
            handle.score(
                name=name,
                value=float(value),
                data_type=ScoreDataType.NUMERIC,
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


def _score_type(
    value: float | bool,
) -> Literal[ScoreDataType.NUMERIC, ScoreDataType.BOOLEAN]:
    return ScoreDataType.BOOLEAN if isinstance(value, bool) else ScoreDataType.NUMERIC


def _score_value(value: float | bool) -> float | bool:
    if isinstance(value, bool):
        return value
    return float(value)


def _metadata_value(value: float | bool | None) -> float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _emit_component_state_scores(
    handle: ObservationHandle | None,
    *,
    score_name: str,
    state: ScoreComponentTelemetryState,
    operation: str,
    comment: str | None = None,
) -> None:
    if state.available and state.value is not None:
        _emit_score_for_handle(
            handle,
            name=score_name,
            value=_score_value(state.value),
            data_type=_score_type(state.value),
            comment=comment,
            operation=operation,
        )
    _emit_score_for_handle(
        handle,
        name=f"{score_name}_available",
        value=state.available,
        data_type=ScoreDataType.BOOLEAN,
        comment=comment,
        operation=f"{operation}.availability",
    )


def _component_state_metadata(
    component_states: Mapping[str, ScoreComponentTelemetryState],
) -> dict[str, dict[str, object | None]]:
    return {
        name: {
            "available": state.available,
            "value": _metadata_value(state.value),
            "reason": state.reason,
        }
        for name, state in component_states.items()
    }


def emit_localization_task_telemetry(
    handle: ObservationHandle | None,
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

    emitted_score_names: list[str] = []
    for component_name in LOCALIZATION_CANONICAL_COMPONENTS:
        state = artifact.canonical_component_states[component_name]
        score_name = f"{score_prefix}.{component_name}"
        _emit_component_state_scores(
            handle,
            score_name=score_name,
            state=state,
            operation="localization_task.score",
        )
        emitted_score_names.append(f"{score_name}_available")
        if state.available and state.value is not None:
            emitted_score_names.append(score_name)
    for metric_name, value in artifact.metric_map.items():
        if metric_name in LOCALIZATION_CANONICAL_COMPONENTS:
            continue
        score_name = f"{score_prefix}.{metric_name}"
        _emit_score_for_handle(
            handle,
            name=score_name,
            value=_score_value(value),
            data_type=_score_type(value),
            operation="localization_task.score",
        )
        emitted_score_names.append(score_name)

    metadata: dict[str, object] = {
        "telemetry": dict(telemetry_payload),
        "score_summary": artifact.score_summary.model_dump(mode="json"),
        "score_context": artifact.compact_score_context.model_dump(mode="json", exclude_none=True),
        "score_components": _component_state_metadata(
            artifact.canonical_component_states
        ),
        "predicted_files": list(predicted_files),
        "changed_files": list(changed_files),
        "score_metric_names": sorted(emitted_score_names),
    }
    if task_summary is not None:
        metadata["task_summary"] = dict(task_summary)
    _update_observation_metadata(
        handle,
        metadata,
        operation="localization_task.metadata",
        observation_name="localization_task",
    )


def emit_optimize_iteration_telemetry(
    handle: ObservationHandle | None,
    *,
    metrics: EvaluationMetrics,
    score_component_states: Mapping[str, ScoreComponentTelemetryState] | None = None,
    iteration: int,
    status: str | None = None,
    pipeline_passed: bool | None = None,
    score_context_summary: Mapping[str, object] | None = None,
    error: str | None = None,
) -> None:
    component_states = optimize_component_states(
        metrics,
        score_component_states=score_component_states,
        pipeline_passed=pipeline_passed,
    )
    comment = json.dumps({"iteration": iteration})
    emitted_score_names: list[str] = []
    for metric_name, state in component_states.items():
        _emit_component_state_scores(
            handle,
            score_name=metric_name,
            state=state,
            comment=comment,
            operation="optimize_iteration.score",
        )
        emitted_score_names.append(f"{metric_name}_available")
        if state.available and state.value is not None:
            emitted_score_names.append(metric_name)
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
        "evaluation_metric_names": sorted(emitted_score_names),
        "evaluation": context,
        "score_components": _component_state_metadata(component_states),
    }
    if score_context_summary:
        metadata["score_context_summary"] = dict(score_context_summary)
    if error:
        metadata["error"] = error
    _update_observation_metadata(
        handle,
        {key: value for key, value in metadata.items() if value is not None},
        operation="optimize_iteration.metadata",
        observation_name="optimize_iteration",
    )


__all__ = [
    "emit_localization_task_telemetry",
    "emit_optimize_iteration_telemetry",
]
