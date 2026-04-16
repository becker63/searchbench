from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

from harness.scoring import BatchScoreSummary, ScoreBundle

_STRICT_DEBUG = bool(os.getenv("LANGFUSE_STRICT_DEBUG"))
ScoreEmitFn: TypeAlias = Callable[..., None]


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


def _handle_trace_id(handle: object | None) -> str | None:
    trace_id = getattr(handle, "trace_id", None) if handle is not None else None
    return trace_id if isinstance(trace_id, str) and trace_id else None


def _handle_observation_id(handle: object | None) -> str | None:
    observation_id = getattr(handle, "id", None) if handle is not None else None
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
        logging.debug("Langfuse score emission skipped: %s", exc)
        return
    try:
        create_fn = client.create_score
    except AttributeError as exc:
        if _STRICT_DEBUG:
            raise RuntimeError("Langfuse client missing create_score API") from exc
        logging.debug("Langfuse score emission unavailable: missing create_score API")
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
        logging.debug("Langfuse score payload invalid; skipping score %s: %s", name, exc)
        return
    cleaned = payload.model_dump(exclude_none=True)
    try:
        create_fn(**cleaned)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        if _STRICT_DEBUG:
            raise
        logging.debug("Langfuse score emission failed for %s: %s", name, exc)
        return


def emit_score_for_handle(
    handle: object | None,
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
        score_fn = getattr(handle, "score", None)
        if callable(score_fn):
            try:
                score_fn(  # type: ignore[call-arg]
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
                logging.debug("Langfuse handle.score failed for %s: %s", name, exc)
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
    handle: object | None,
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
        score_trace_fn = getattr(handle, "score_trace", None)
        if callable(score_trace_fn):
            try:
                score_trace_fn(  # type: ignore[call-arg]
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
                logging.debug("Langfuse handle.score_trace failed for %s: %s", name, exc)
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
    handle: object | None,
    *,
    name: str,
    value: float,
    emit_fn: ScoreEmitFn | None,
) -> None:
    emitter = emit_fn or emit_score_for_handle
    emitter(
        handle,
        name=name,
        value=float(value),
        data_type="NUMERIC",
    )


def _safe_update_metadata(handle: object | None, metadata: Mapping[str, object]) -> None:
    updater = getattr(handle, "update", None)
    if callable(updater):
        try:
            updater(metadata=dict(metadata))
        except Exception:
            if _STRICT_DEBUG:
                raise
            logging.debug("Langfuse score metadata update skipped", exc_info=True)


def score_bundle_metadata(
    bundle: ScoreBundle,
    *,
    metadata_key: str = "score_bundle",
) -> dict[str, object]:
    hidden_components: list[str] = []
    unavailable_components: dict[str, object] = {}
    for component_name, result in bundle.results.items():
        stable_name = result.name or component_name
        if not result.visible_to_agent:
            hidden_components.append(stable_name)
            continue
        if not result.available or not isinstance(result.value, (int, float)):
            unavailable_components[stable_name] = result.reason or "unavailable"
    return {
        metadata_key: bundle.model_dump(mode="json"),
        f"{metadata_key}_diagnostics": {
            "hidden_components": hidden_components,
            "unavailable_components": unavailable_components,
            "available": bundle.available,
        },
    }


def emit_score_bundle(
    handle: object | None,
    bundle: ScoreBundle,
    *,
    prefix: str,
    metadata_key: str = "score_bundle",
    emit_fn: ScoreEmitFn | None = None,
) -> None:
    """
    Emit the canonical per-item scoring surface.

    Ownership convention: callers attach score bundles to the evaluated task span,
    not raw backend spans. Backend spans may still emit auxiliary diagnostics such
    as node counts, but only the evaluated task span has the gold-aware
    `ScoreBundle` needed for composed and component scores.
    """
    if bundle.composed_score is not None:
        _emit_numeric_score(
            handle,
            name=f"{prefix}.composed_score",
            value=float(bundle.composed_score),
            emit_fn=emit_fn,
        )

    for component_name, result in bundle.results.items():
        stable_name = result.name or component_name
        if not result.visible_to_agent:
            continue
        if not result.available or not isinstance(result.value, (int, float)):
            continue
        _emit_numeric_score(
            handle,
            name=f"{prefix}.{stable_name}",
            value=float(result.value),
            emit_fn=emit_fn,
        )

    _safe_update_metadata(handle, score_bundle_metadata(bundle, metadata_key=metadata_key))


def emit_task_score_summary(
    handle: object | None,
    summary: ScoreBundle,
    *,
    prefix: str,
    metadata_key: str = "score_summary",
    emit_fn: ScoreEmitFn | None = None,
) -> None:
    """Emit a task score summary using the same names as a concrete score bundle."""
    emit_score_bundle(
        handle,
        summary,
        prefix=prefix,
        metadata_key=metadata_key,
        emit_fn=emit_fn,
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
    emit_fn: ScoreEmitFn | None = None,
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
    "emit_score_bundle",
    "emit_score_for_handle",
    "emit_task_score_summary",
    "emit_trace_score_for_handle",
    "score_bundle_metadata",
]
