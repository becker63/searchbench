from __future__ import annotations

import logging
import os
from collections.abc import Callable

from langfuse import LangfuseGeneration, LangfuseSpan
from pydantic import BaseModel, ConfigDict, model_validator

from harness.scoring import BatchScoreSummary

_STRICT_DEBUG = bool(os.getenv("LANGFUSE_STRICT_DEBUG"))


class ScorePayload(BaseModel):
    """Strict payload for Langfuse score emission."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | int | bool | str
    data_type: str | None = None
    trace_id: str | None = None
    observation_id: str | None = None
    session_id: str | None = None
    config_id: str | None = None
    comment: str | None = None
    score_id: str | None = None

    @model_validator(mode="after")
    def _require_identifier(self) -> "ScorePayload":
        if not any([self.trace_id, self.observation_id, self.session_id]):
            raise ValueError("Score emission requires a trace, observation, or session identifier")
        return self


def _infer_data_type(value: float | int | bool | str, provided: str | None) -> str | None:
    if provided is not None:
        return provided
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, (int, float)):
        return "NUMERIC"
    if isinstance(value, str):
        return "CATEGORICAL"
    return None


def _normalize_score_value(
    value: float | int | bool | str,
    data_type: str | None,
) -> tuple[float | int | str, str | None]:
    resolved_type = _infer_data_type(value, data_type)
    if isinstance(value, bool):
        return (1 if value else 0), resolved_type
    return value, resolved_type


def _handle_trace_id(handle: LangfuseSpan | LangfuseGeneration | None) -> str | None:
    if handle is None:
        return None
    trace_id = handle.trace_id
    return trace_id if isinstance(trace_id, str) and trace_id else None


def _handle_observation_id(handle: LangfuseSpan | LangfuseGeneration | None) -> str | None:
    if handle is None:
        return None
    observation_id = handle.id
    return observation_id if isinstance(observation_id, str) and observation_id else None


def emit_score(
    *,
    name: str,
    value: float | int | bool | str,
    data_type: str | None = None,
    trace_id: str | None = None,
    observation_id: str | None = None,
    session_id: str | None = None,
    config_id: str | None = None,
    comment: str | None = None,
    score_id: str | None = None,
) -> None:
    """Submit a score via Langfuse SDK using hosted run identifiers."""
    from . import get_langfuse_client  # local import to avoid circularity
    from . import ensure_langfuse_auth  # avoid circularity lint

    normalized_value, resolved_type = _normalize_score_value(value, data_type)
    try:
        client = get_langfuse_client()
        ensure_langfuse_auth()
    except Exception as exc:  # noqa: BLE001
        logging.warning("Langfuse score emission skipped score=%s error=%s", name, exc)
        return
    try:
        create_fn = client.create_score
    except AttributeError as exc:
        if _STRICT_DEBUG:
            raise RuntimeError("Langfuse client missing create_score API") from exc
        logging.warning("Langfuse score emission unavailable score=%s error=missing_create_score_api", name)
        return
    try:
        payload = ScorePayload(
            name=name,
            value=normalized_value,
            data_type=resolved_type,
            trace_id=trace_id,
            observation_id=observation_id,
            session_id=session_id,
            config_id=config_id,
            comment=comment,
            score_id=score_id,
        )
    except Exception as exc:  # noqa: BLE001
        if _STRICT_DEBUG:
            raise
        logging.warning("Langfuse score payload invalid score=%s error=%s", name, exc)
        return
    cleaned = payload.model_dump(exclude_none=True)
    try:
        create_fn(**cleaned)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        if _STRICT_DEBUG:
            raise
        logging.warning("Langfuse score emission failed score=%s error=%s", name, exc)
        return


def emit_score_for_handle(
    handle: LangfuseSpan | LangfuseGeneration | None,
    *,
    name: str,
    value: float | int | bool | str,
    data_type: str | None = None,
    session_id: str | None = None,
    config_id: str | None = None,
    comment: str | None = None,
    score_id: str | None = None,
) -> None:
    """Emit a score using IDs derived from a Langfuse trace/span handle."""
    normalized_value, resolved_type = _normalize_score_value(value, data_type)
    # Prefer the SDK surface on the active observation when possible. The Python
    # SDK docs show score_id only on create_score(), so use the low-level path
    # when callers need idempotency.
    if handle is not None and score_id is None:
        try:
            handle.score(
                name=name,
                value=normalized_value,
                data_type=resolved_type,
                comment=comment,
                config_id=config_id,
            )
            return
        except Exception as exc:  # noqa: BLE001
            if _STRICT_DEBUG:
                raise
            logging.warning(
                "Langfuse handle.score failed observation=%s score=%s error=%s",
                _handle_observation_id(handle) or type(handle).__name__,
                name,
                exc,
            )
    # Fall back to explicit identifier-based emission only when the handle
    # exposes the real trace id. Langfuse requires trace_id for trace and
    # observation scores; handle.id is an observation id and must not be reused
    # as a fabricated trace id.
    trace_id = _handle_trace_id(handle)
    if trace_id is None and session_id is None:
        logging.debug(
            "Langfuse score emission skipped for %s: missing trace_id on handle",
            name,
        )
        return
    observation_id = _handle_observation_id(handle)
    emit_score(
        name=name,
        value=normalized_value,
        data_type=resolved_type,
        trace_id=trace_id,
        observation_id=observation_id,
        session_id=session_id,
        config_id=config_id,
        comment=comment,
        score_id=score_id,
    )


def emit_trace_score_for_handle(
    handle: LangfuseSpan | LangfuseGeneration | None,
    *,
    name: str,
    value: float | int | bool | str,
    data_type: str | None = None,
    session_id: str | None = None,
    config_id: str | None = None,
    comment: str | None = None,
    score_id: str | None = None,
) -> None:
    """Emit a trace-level score using the v4 score API."""
    normalized_value, resolved_type = _normalize_score_value(value, data_type)
    if handle is not None and score_id is None:
        try:
            handle.score_trace(
                name=name,
                value=normalized_value,
                data_type=resolved_type,
                comment=comment,
                config_id=config_id,
            )
            return
        except Exception as exc:  # noqa: BLE001
            if _STRICT_DEBUG:
                raise
            logging.warning(
                "Langfuse handle.score_trace failed observation=%s score=%s error=%s",
                _handle_observation_id(handle) or type(handle).__name__,
                name,
                exc,
            )
    trace_id = _handle_trace_id(handle)
    if trace_id is None and session_id is None:
        logging.debug(
            "Langfuse trace score emission skipped for %s: missing trace_id on handle",
            name,
        )
        return
    emit_score(
        name=name,
        value=normalized_value,
        data_type=resolved_type,
        trace_id=trace_id,
        observation_id=None,
        session_id=session_id,
        config_id=config_id,
        comment=comment,
        score_id=score_id,
    )


def _emit_numeric_score(
    handle: LangfuseSpan | LangfuseGeneration | None,
    *,
    name: str,
    value: float,
    emit_fn: Callable[..., None] | None,
) -> None:
    emitter = emit_fn or emit_score_for_handle
    emitter(
        handle,
        name=name,
        value=float(value),
        data_type="NUMERIC",
    )


def _safe_update_metadata(handle: LangfuseSpan | LangfuseGeneration | None, metadata: dict[str, object]) -> None:
    if handle is None:
        return
    try:
        handle.update(metadata=dict(metadata))
    except Exception as exc:  # noqa: BLE001
        if _STRICT_DEBUG:
            raise
        logging.warning(
            "Langfuse batch score metadata update failed observation=%s keys=%s error=%s",
            _handle_observation_id(handle) or type(handle).__name__,
            sorted(str(key) for key in metadata),
            exc,
        )


def batch_score_summary_metadata(
    summary: BatchScoreSummary,
    *,
    metadata_key: str = "score_summary",
) -> dict[str, object]:
    missing_components: dict[str, int] = {}
    hidden_components: list[str] = []
    for component_name, component in summary.components.items():
        stable_name = component.name or component_name
        if not component.visible_to_agent:
            hidden_components.append(stable_name)
            continue
        if component.missing_count:
            missing_components[stable_name] = component.missing_count
    return {
        metadata_key: summary.model_dump(mode="json"),
        f"{metadata_key}_diagnostics": {
            "hidden_components": hidden_components,
            "missing_components": missing_components,
            "composed_missing_count": summary.composed_missing_count,
        },
    }


def emit_batch_score_summary(
    handle: object | None,
    summary: BatchScoreSummary,
    *,
    prefix: str,
    metadata_key: str = "score_summary",
    emit_fn: Callable[..., None] | None = None,
) -> None:
    """Emit the canonical dataset/root scoring surface for an aggregate summary."""
    score_fn = emit_fn or emit_trace_score_for_handle
    if summary.aggregate_composed_score is not None:
        _emit_numeric_score(
            handle,
            name=f"{prefix}.aggregate_composed_score",
            value=float(summary.aggregate_composed_score),
            emit_fn=score_fn,
        )

    for component_name, component in summary.components.items():
        stable_name = component.name or component_name
        if not component.visible_to_agent:
            continue
        if component.average is None:
            continue
        _emit_numeric_score(
            handle,
            name=f"{prefix}.aggregate_{stable_name}",
            value=float(component.average),
            emit_fn=score_fn,
        )

    _safe_update_metadata(
        handle,
        batch_score_summary_metadata(summary, metadata_key=metadata_key),
    )


__all__ = [
    "ScorePayload",
    "batch_score_summary_metadata",
    "emit_batch_score_summary",
    "emit_score",
    "emit_score_for_handle",
    "emit_trace_score_for_handle",
]
