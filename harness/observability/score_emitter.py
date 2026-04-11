from __future__ import annotations

import logging
import os

from pydantic import BaseModel, ConfigDict, model_validator

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
    from .langfuse import get_langfuse_client  # local import to avoid circularity
    from .langfuse import ensure_langfuse_auth  # avoid circularity lint

    resolved_type = _infer_data_type(value, data_type)
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
            value=value,
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
    # Prefer the SDK surface on the active observation when possible.
    if handle is not None:
        try:
            handle.score(  # type: ignore[call-arg]
                name=name,
                value=value,
                data_type=data_type,
                comment=comment,
                config_id=config_id,
                score_id=score_id,
            )
            return
        except AttributeError:
            pass
        except Exception as exc:  # noqa: BLE001
            if _STRICT_DEBUG:
                raise
            logging.debug("Langfuse handle.score failed for %s: %s", name, exc)
    # Fall back to explicit identifier-based emission (for handles without score()).
    trace_id = None
    observation_id = None
    try:
        trace_id = handle.trace_id  # type: ignore[attr-defined]
    except Exception:
        try:
            trace_id = handle.id  # type: ignore[attr-defined]
        except Exception:
            trace_id = None
    try:
        observation_id = handle.id  # type: ignore[attr-defined]
    except Exception:
        observation_id = None
    emit_score(
        name=name,
        value=value,
        data_type=data_type,
        trace_id=trace_id,
        observation_id=observation_id,
        session_id=session_id,
        config_id=config_id,
        comment=comment,
        score_id=score_id,
    )


__all__ = ["emit_score", "emit_score_for_handle", "ScorePayload"]
