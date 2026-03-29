from __future__ import annotations

from harness.observability import langfuse as lf


class DummySpan:
    def __init__(self):
        self.scores: list[tuple[str, float, dict[str, object] | None]] = []

    def score(self, name: str, value: float, metadata=None):
        self.scores.append((name, value, metadata))


def test_langfuse_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_ENABLED", "0")
    assert lf.get_langfuse_client() is None


def test_record_score_passes_metadata():
    span = DummySpan()
    lf.record_score(span, "metric", 1.0, metadata={"a": 1})
    assert span.scores == [("metric", 1.0, {"a": 1})]
