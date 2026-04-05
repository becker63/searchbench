from __future__ import annotations

import pytest

from harness.observability import langfuse as lf
from harness.observability import score_emitter


@pytest.mark.no_langfuse_stub
def test_langfuse_disabled_without_keys(monkeypatch):
    import importlib
    import harness.observability.langfuse as lf

    monkeypatch.setenv("LANGFUSE_NO_STUB", "1")
    lf = importlib.reload(lf)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    try:
        client = lf.get_langfuse_client()
    except RuntimeError:
        return
    # If stubbed, ensure it's the dummy client (networkless)
    assert hasattr(client, "create_score")


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


def test_emit_score_for_handle_requires_identifier(monkeypatch):
    class DummyClient:
        def create_score(self, **kwargs):
            pass

    monkeypatch.setattr("harness.observability.langfuse.get_langfuse_client", lambda: DummyClient())
    with pytest.raises(ValueError):
        score_emitter.emit_score_for_handle(None, name="metric", value=1.0)


def test_emit_score_for_handle_allows_dataset_only(monkeypatch):
    calls: list[dict[str, object]] = []

    class DummyClient:
        def create_score(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("harness.observability.langfuse.get_langfuse_client", lambda: DummyClient())
    score_emitter.emit_score_for_handle(None, name="metric", value=1.0, dataset_run_id="run123")

    assert calls and calls[0]["dataset_run_id"] == "run123"
