from __future__ import annotations

import hashlib
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


__all__ = ["SessionScope", "SessionConfig", "resolve_session_id"]
