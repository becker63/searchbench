from __future__ import annotations

import pytest

from harness.observability import langfuse as lf
from harness.observability import score_emitter


def test_langfuse_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError):
        lf.get_langfuse_client()


def test_emit_score_uses_client(monkeypatch):
    calls: list[dict[str, object]] = []

    class DummyClient:
        def create_score(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("harness.observability.langfuse.get_langfuse_client", lambda: DummyClient())
    score_emitter.emit_score(name="metric", value=1.0, trace_id="t1")

    assert calls
    emitted = calls[0]
    assert emitted["name"] == "metric"
    assert emitted["value"] == 1.0
    assert emitted.get("trace_id") == "t1"
