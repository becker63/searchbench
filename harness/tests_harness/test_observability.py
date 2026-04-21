from __future__ import annotations

import pytest

from harness.telemetry import tracing as lf
from harness.telemetry.tracing import score_emitter


@pytest.mark.no_langfuse_stub
def test_langfuse_disabled_without_keys(monkeypatch):
    import importlib
    import harness.telemetry.tracing as lf

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

    monkeypatch.setattr("harness.telemetry.tracing.get_langfuse_client", lambda: DummyClient())
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

    monkeypatch.setattr("harness.telemetry.tracing.get_langfuse_client", lambda: DummyClient())
    score_emitter.emit_score_for_handle(None, name="metric", value=1.0)


def test_emit_score_for_handle_allows_dataset_only(monkeypatch):
    calls: list[dict[str, object]] = []

    class DummyClient:
        def create_score(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("harness.telemetry.tracing.get_langfuse_client", lambda: DummyClient())
    score_emitter.emit_score_for_handle(None, name="metric", value=1.0, session_id="sess123")

    assert calls
    emitted = calls[0]
    assert emitted.get("session_id") == "sess123"


def test_start_observation_accepts_context_manager(monkeypatch):
    class CMObservation:
        def __init__(self):
            self.id = "cm"
            self.trace_id = "trace"
            self.exited = 0
            self.end_called = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.exited += 1
            return False

        def end(self):
            self.end_called += 1

    monkeypatch.setattr(
        "harness.telemetry.tracing.get_langfuse_client",
        lambda: type("C", (), {"start_as_current_observation": lambda *a, **k: CMObservation()})(),
    )

    with lf.start_observation(name="root") as obs:
        assert isinstance(obs, CMObservation)
        assert obs.id == "cm"
    assert obs.exited == 1
    assert obs.end_called == 0


def test_start_child_observation_uses_langfuse_child_context_manager():
    class PlainChild:
        def __init__(self, session_id: str | None):
            self.id = "child"
            self.name = "child"
            self.session_id = session_id
            self.exited = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.exited += 1
            return False

    class PlainParent:
        def __init__(self, session_id: str | None):
            self.session_id = session_id
            self.started = 0

        def start_as_current_observation(self, **kwargs):
            self.started += 1
            meta = kwargs.get("metadata") or {}
            child_session = meta.get("session_id") or self.session_id
            return PlainChild(child_session)

    parent = PlainParent("parent-session")

    with lf.start_observation(name="child", parent=parent) as child:
        assert isinstance(child, PlainChild)
        assert child.session_id == "parent-session"
        assert child.exited == 0

    assert parent.started == 1
    assert child.exited == 1


def test_emit_score_best_effort_when_client_missing(monkeypatch):
    monkeypatch.setattr(
        "harness.telemetry.tracing.get_langfuse_client",
        lambda: (_ for _ in ()).throw(RuntimeError("no client")),
    )
    # Should not raise
    score_emitter.emit_score(name="metric", value=1.0, trace_id="t1")


def test_emit_score_handles_missing_api(monkeypatch):
    class DummyClient:
        api = type("A", (), {})()

    monkeypatch.setattr("harness.telemetry.tracing.get_langfuse_client", lambda: DummyClient())
    # Should not raise even though no create_score API
    score_emitter.emit_score(name="metric", value=1.0, trace_id="t1")


def test_propagate_context_drops_invalid_metadata(monkeypatch, caplog):
    captured: dict[str, object] = {}

    def fake_propagate_attributes(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(lf, "propagate_attributes", fake_propagate_attributes)
    caplog.set_level("WARNING")

    lf.propagate_context(
        metadata={
            "source": "api",
            "task_identity": "task-1",
            "repo": "owner/repo",
            "toolong": "x" * 400,
        }
    )

    meta = captured.get("metadata") or {}
    assert isinstance(meta, dict)
    assert meta == {"source": "api"}
    assert "task_identity" not in meta
    assert "repo" not in meta
    assert "toolong" not in meta
    assert "propagated_metadata_key_dropped" in caplog.text


def test_propagate_context_strict_rejects_invalid_key(monkeypatch):
    monkeypatch.setenv("LANGFUSE_STRICT_DEBUG", "1")
    import importlib

    import harness.telemetry.tracing as lf_mod

    importlib.reload(lf_mod)
    with pytest.raises(ValueError):
        lf_mod.propagate_context(metadata={"task_identity": "task-1"})


def test_observation_metadata_preserves_nested_json():
    metadata = lf.serialize_observation_metadata(
        {
            "score_context": {"gold_min_hops": 0, "flags": [True, False]},
            "count": 1,
        }
    )

    assert metadata == {
        "score_context": {"gold_min_hops": 0, "flags": [True, False]},
        "count": 1,
    }
