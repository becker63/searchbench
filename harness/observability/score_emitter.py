from __future__ import annotations

import json
from typing import Any, Mapping, cast

from pydantic import BaseModel, ConfigDict


class ScorePayload(BaseModel):
    """Strict payload for Langfuse score emission."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | int | bool | str
    data_type: str | None = None
    trace_id: str | None = None
    observation_id: str | None = None
    dataset_run_id: str | None = None
    session_id: str | None = None
    config_id: str | None = None
    comment: str | None = None
    score_id: str | None = None


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


def emit_score(
    *,
    name: str,
    value: float | int | bool | str,
    data_type: str | None = None,
    trace_id: str | None = None,
    observation_id: str | None = None,
    dataset_run_id: str | None = None,
    session_id: str | None = None,
    config_id: str | None = None,
    comment: str | None = None,
    score_id: str | None = None,
) -> None:
    """Submit a score via Langfuse SDK using hosted run identifiers."""
    if not any([trace_id, observation_id, dataset_run_id, session_id]):
        raise RuntimeError("Score emission requires at least one identifier (trace/observation/datasetRun/session)")
    from .langfuse import get_langfuse_client  # local import to avoid circularity

    resolved_type = _infer_data_type(value, data_type)
    client = get_langfuse_client()
    if not hasattr(client, "create_score"):
        raise RuntimeError("Langfuse client unavailable for scoring")
    payload = ScorePayload(
        name=name,
        value=value,
        data_type=resolved_type,
        trace_id=trace_id,
        observation_id=observation_id,
        dataset_run_id=dataset_run_id,
        session_id=session_id,
        config_id=config_id,
        comment=comment,
        score_id=score_id,
    )
    cleaned = payload.model_dump(exclude_none=True)
    client_any = cast(Any, client)
    client_any.create_score(**cleaned)  # type: ignore[arg-type]


def emit_score_for_handle(
    handle: object | None,
    *,
    name: str,
    value: float | int | bool | str,
    data_type: str | None = None,
    dataset_run_id: str | None = None,
    session_id: str | None = None,
    config_id: str | None = None,
    comment: str | None = None,
    score_id: str | None = None,
) -> None:
    """Emit a score using IDs derived from a Langfuse trace/span handle."""
    if handle is None:
        raise RuntimeError("Langfuse handle is required to emit score")
    trace_id = getattr(handle, "trace_id", None) or getattr(handle, "id", None)
    observation_id = getattr(handle, "id", None)
    emit_score(
        name=name,
        value=value,
        data_type=data_type,
        trace_id=trace_id,
        observation_id=observation_id,
        dataset_run_id=dataset_run_id,
        session_id=session_id,
        config_id=config_id,
        comment=comment,
        score_id=score_id,
    )


__all__ = ["emit_score", "emit_score_for_handle", "ScorePayload"]
