from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

SessionScope = Literal["run", "item"]


class SessionConfig(BaseModel):
    """Session policy at public entrypoints."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    session_scope: SessionScope = "run"
    allow_generated_fallback: bool = True


def _derive_id(*parts: str) -> str:
    joined = "::".join(p for p in parts if p)
    digest = hashlib.sha256(joined.encode()).hexdigest()[:12]
    return f"session-{digest}"


def _slug(value: str, *, limit: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug[:limit].strip("-") or "item"


def derive_optimize_writer_session_id(
    *,
    optimize_run_id: str,
    task_identity: str,
    ordinal: int,
) -> str:
    """
    Build the replay session id for one optimize writer loop.

    Optimize run correlation is metadata-only. This id is intentionally scoped
    to the writer/pipeline loop for one task so Langfuse session replay tells
    the policy-writing story instead of absorbing evaluator/localizer traces.
    """
    digest = hashlib.sha256(f"{optimize_run_id}::{task_identity}::{ordinal}".encode()).hexdigest()[:10]
    run_slug = _slug(optimize_run_id, limit=32)
    return f"writer:{run_slug}:{ordinal}:{digest}"


def resolve_session_id(
    config: SessionConfig | None,
    *,
    run_id: str | None = None,
    item_id: str | None = None,
) -> str | None:
    """
    Resolve the effective session id at the entrypoint boundary.

    Precedence:
    1) Explicit session_id in config
    2) If allow_generated_fallback is True, derive based on scope and context
    3) Otherwise None
    """
    if config is None:
        return None
    if config.session_id:
        return config.session_id
    if not config.allow_generated_fallback:
        return None
    if config.session_scope == "item" and item_id:
        return _derive_id(run_id or "", item_id)
    if config.session_scope == "run" and run_id:
        return _derive_id(run_id)
    return None


__all__ = [
    "SessionScope",
    "SessionConfig",
    "derive_optimize_writer_session_id",
    "resolve_session_id",
]
