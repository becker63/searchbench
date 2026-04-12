from __future__ import annotations

from harness.telemetry.tracing import score_emitter


def test_emit_score_prefers_handle_score():
    called: list[dict[str, object]] = []

    class Dummy:
        id = "obs"
        trace_id = "trace"

        def score(self, **kwargs):
            called.append(kwargs)

    score_emitter.emit_score_for_handle(Dummy(), name="metric", value=1.0, score_id="sid")
    assert called
    assert called[0]["name"] == "metric"
    assert called[0]["score_id"] == "sid"
