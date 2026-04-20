from __future__ import annotations

from harness.scoring import (
    BatchScoreSummary,
    ComponentAggregate,
    ComposeMode,
    Goal,
    ScoreBundle,
    ScoreResult,
)
from harness.telemetry.tracing import score_emitter


def test_emit_score_prefers_handle_score_without_idempotency_key():
    called: list[dict[str, object]] = []

    class Dummy:
        id = "obs"
        trace_id = "trace"

        def score(self, **kwargs):
            called.append(kwargs)

    score_emitter.emit_score_for_handle(Dummy(), name="metric", value=1.0)
    assert called
    assert called[0]["name"] == "metric"
    assert "score_id" not in called[0]


def test_emit_score_for_handle_uses_create_score_when_score_id_is_needed(monkeypatch):
    handle_calls: list[dict[str, object]] = []
    client_calls: list[dict[str, object]] = []

    class DummyHandle:
        id = "obs"
        trace_id = "trace"

        def score(self, **kwargs):
            handle_calls.append(kwargs)

    class DummyClient:
        def auth_check(self):
            return True

        def create_score(self, **kwargs):
            client_calls.append(kwargs)

    monkeypatch.setattr(
        "harness.telemetry.tracing.get_langfuse_client", lambda: DummyClient()
    )

    score_emitter.emit_score_for_handle(
        DummyHandle(),
        name="metric",
        value=True,
        data_type="BOOLEAN",
        score_id="sid",
        config_id="cfg",
        comment="ok",
    )

    assert handle_calls == []
    assert client_calls == [
        {
            "name": "metric",
            "value": 1,
            "data_type": "BOOLEAN",
            "trace_id": "trace",
            "observation_id": "obs",
            "config_id": "cfg",
            "comment": "ok",
            "score_id": "sid",
        }
    ]


def test_emit_score_for_handle_falls_back_when_handle_score_raises(monkeypatch):
    client_calls: list[dict[str, object]] = []

    class DummyHandle:
        id = "obs"
        trace_id = "trace"

        def score(self, **kwargs):
            raise RuntimeError("queue_exceeded")

    class DummyClient:
        def auth_check(self):
            return True

        def create_score(self, **kwargs):
            client_calls.append(kwargs)

    monkeypatch.setattr(
        "harness.telemetry.tracing.get_langfuse_client", lambda: DummyClient()
    )

    score_emitter.emit_score_for_handle(DummyHandle(), name="metric", value=1.0)

    assert client_calls
    assert client_calls[0]["trace_id"] == "trace"
    assert client_calls[0]["observation_id"] == "obs"


def test_emit_trace_score_for_handle_prefers_score_trace():
    called: list[dict[str, object]] = []

    class Dummy:
        id = "trace"

        def score(self, **kwargs):
            raise AssertionError("observation score should not be used")

        def score_trace(self, **kwargs):
            called.append(kwargs)

    score_emitter.emit_trace_score_for_handle(Dummy(), name="aggregate", value=0.5)

    assert called == [
        {
            "name": "aggregate",
            "value": 0.5,
            "data_type": "NUMERIC",
            "comment": None,
            "config_id": None,
        }
    ]


def test_emit_batch_score_summary_emits_visible_component_averages():
    emitted: list[dict[str, object]] = []

    class Dummy:
        id = "trace"

        def score_trace(self, **kwargs):
            emitted.append(kwargs)

    bundle = ScoreBundle(
        results={
            "gold_hop": ScoreResult(
                name="gold_hop",
                value=1.0,
                goal=Goal.MAXIMIZE,
                available=True,
                visible_to_agent=True,
            )
        },
        composed_score=1.0,
        compose_mode=ComposeMode.WEIGHTED_SUM,
        compose_weights={"gold_hop": 1.0},
        available=True,
    )
    summary = BatchScoreSummary(
        aggregate_composed_score=1.0,
        composed_available_count=1,
        composed_missing_count=0,
        total_count=1,
        items={"task": bundle},
        components={
            "gold_hop": ComponentAggregate(
                name="gold_hop",
                average=1.0,
                available_count=1,
                missing_count=0,
                total_count=1,
                visible_to_agent=True,
            )
        },
    )

    score_emitter.emit_batch_score_summary(
        Dummy(),
        summary,
        prefix="baseline",
    )

    assert [item["name"] for item in emitted] == [
        "baseline.aggregate_composed_score",
        "baseline.aggregate_gold_hop",
    ]
