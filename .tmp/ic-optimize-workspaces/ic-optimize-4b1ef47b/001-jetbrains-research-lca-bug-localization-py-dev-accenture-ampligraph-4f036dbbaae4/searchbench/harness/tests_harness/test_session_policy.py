from __future__ import annotations

import pytest

from contextlib import contextmanager
from typing import Any

from harness.telemetry.tracing import start_observation
from harness.telemetry.tracing.score_emitter import emit_score
from harness.telemetry.tracing.session_policy import (
    SessionConfig,
    derive_optimize_writer_session_id,
    resolve_session_id,
)


@contextmanager
def _dummy_observation(session_id: str | None = None):
    class Obs:
        def __init__(self, sid: str | None):
            self.session_id = sid

        def start_observation(self, **kwargs: Any):
            meta = kwargs.get("metadata") or {}
            child_sid = meta.get("session_id") or self.session_id
            if "session_id" in meta and self.session_id and meta["session_id"] != self.session_id:
                raise RuntimeError("session_id conflict")
            return _dummy_observation(child_sid)

    yield Obs(session_id)


def test_child_span_inherits_session(monkeypatch):
    monkeypatch.setattr(
        "harness.telemetry.tracing.get_langfuse_client",
        lambda: type(
            "C",
            (),
            {"start_as_current_observation": lambda *a, **k: _dummy_observation(k.get("metadata", {}).get("session_id"))},
        )(),
    )
    with start_observation(name="root", metadata={"session_id": "root-session"}) as parent:
        with start_observation(name="child", parent=parent) as child:
            assert getattr(child, "session_id", None) == "root-session"


def test_conflicting_child_session_raises(monkeypatch):
    monkeypatch.setattr(
        "harness.telemetry.tracing.get_langfuse_client",
        lambda: type(
            "C",
            (),
            {"start_as_current_observation": lambda *a, **k: _dummy_observation(k.get("metadata", {}).get("session_id"))},
        )(),
    )
    with start_observation(name="root", metadata={"session_id": "root-session"}) as parent:
        with pytest.raises(RuntimeError):
            with start_observation(name="child", parent=parent, metadata={"session_id": "other-session"}):
                pass


def test_session_only_score(monkeypatch):
    calls = []

    class DummyClient:
        def create_score(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.delenv("LANGFUSE_STRICT_DEBUG", raising=False)
    import harness.telemetry.tracing.score_emitter as se
    import harness.telemetry.tracing as lf

    monkeypatch.setattr(se, "_STRICT_DEBUG", False)
    monkeypatch.setattr(lf, "_STRICT_DEBUG", False)
    monkeypatch.setattr("harness.telemetry.tracing.get_langfuse_client", lambda: DummyClient())
    emit_score(name="metric", value=1.0, session_id="sess-only")
    assert calls and calls[0]["session_id"] == "sess-only"


def test_resolve_session_defaults():
    cfg = SessionConfig(session_scope="item", allow_generated_fallback=True)
    sess = resolve_session_id(cfg, run_id="run", item_id="item")
    assert sess is not None
    none_cfg = SessionConfig(session_scope="run", allow_generated_fallback=False)
    assert resolve_session_id(none_cfg, run_id="run") is None


def test_optimize_writer_session_id_is_short_stable_and_writer_scoped():
    first = derive_optimize_writer_session_id(
        optimize_run_id="ic-optimize-abcdef12",
        task_identity="lca/py/dev/owner/repo/abc123",
        ordinal=3,
    )
    second = derive_optimize_writer_session_id(
        optimize_run_id="ic-optimize-abcdef12",
        task_identity="lca/py/dev/owner/repo/abc123",
        ordinal=3,
    )
    other = derive_optimize_writer_session_id(
        optimize_run_id="ic-optimize-abcdef12",
        task_identity="lca/py/dev/owner/repo/abc123",
        ordinal=4,
    )

    assert first == second
    assert first != other
    assert first.startswith("writer:ic-optimize-abcdef12:3:")
    assert len(first) < 200
