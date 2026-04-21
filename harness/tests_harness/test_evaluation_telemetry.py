from __future__ import annotations

import pytest

from harness.orchestration.types import EvaluationMetrics
from harness.scoring import (
    ComposeMode,
    Goal,
    ScoreBundle,
    ScoreContext,
    ScoreEngine,
    ScoreResult,
)
from harness.telemetry import evaluation_emitters
from harness.telemetry.observability_models import (
    EvaluationTelemetryArtifact,
    evaluation_metrics_map,
)


def _bundle() -> ScoreBundle:
    return ScoreBundle(
        results={
            "gold_hop": ScoreResult(
                name="gold_hop",
                value=1.0,
                goal=Goal.MAXIMIZE,
                available=True,
                visible_to_agent=True,
            ),
            "issue_hop": ScoreResult(
                name="issue_hop",
                value=0.5,
                goal=Goal.MAXIMIZE,
                available=True,
                visible_to_agent=True,
            ),
            "token_efficiency": ScoreResult(
                name="token_efficiency",
                value=0.25,
                goal=Goal.MAXIMIZE,
                available=True,
                visible_to_agent=True,
            ),
        },
        composed_score=0.75,
        compose_mode=ComposeMode.WEIGHTED_SUM,
        compose_weights={"gold_hop": 0.75, "issue_hop": 0.25},
        available=True,
    )


def _context() -> ScoreContext:
    return ScoreContext(
        predicted_files=("a.py", "missing.py"),
        gold_files=("a.py",),
        gold_min_hops=0,
        issue_min_hops=2,
        previous_total_token_count=123.0,
        resolved_prediction_nodes={"a.py": ("file:a.py",), "missing.py": ()},
        resolved_gold_nodes={"a.py": ("file:a.py",)},
        unresolved_predictions=("missing.py",),
        unresolved_gold_files=(),
        diagnostics={
            "static_graph_available": True,
            "token_usage_available": True,
            "gold_min_hops_available": True,
            "issue_min_hops_available": True,
            "issue_anchor_count": 3,
            "gold_anchor_count": 1,
        },
    )


def _unavailable_context(
    *,
    static_graph_available: bool = True,
    resolved_prediction_count: int = 1,
    resolved_gold_count: int = 1,
    issue_anchor_count: int = 1,
    token_usage_available: bool = True,
) -> ScoreContext:
    return ScoreContext(
        predicted_files=("a.py",),
        gold_files=("b.py",),
        gold_min_hops=None,
        issue_min_hops=None,
        previous_total_token_count=None,
        diagnostics={
            "static_graph_available": static_graph_available,
            "resolved_prediction_count": resolved_prediction_count,
            "resolved_gold_count": resolved_gold_count,
            "issue_anchor_count": issue_anchor_count,
            "token_usage_available": token_usage_available,
        },
    )


def test_evaluation_telemetry_artifact_contains_compact_score_context():
    artifact = EvaluationTelemetryArtifact.from_scoring(
        score_bundle=_bundle(),
        score_context=_context(),
    )

    assert artifact.metric_map == {
        "composed_score": 0.75,
        "gold_hop": 1.0,
        "issue_hop": 0.5,
        "token_efficiency": 0.25,
    }
    compact = artifact.compact_score_context.model_dump(mode="json", exclude_none=True)
    assert compact["gold_min_hops"] == 0
    assert compact["issue_min_hops"] == 2
    assert compact["previous_total_token_count"] == 123.0
    assert compact["static_graph_available"] is True
    assert compact["token_usage_available"] is True
    assert compact["unresolved_prediction_count"] == 1


def test_evaluation_telemetry_artifact_preserves_unavailable_component_state():
    bundle = ScoreEngine().evaluate(
        _unavailable_context(
            static_graph_available=True,
            resolved_prediction_count=0,
            resolved_gold_count=1,
            issue_anchor_count=1,
            token_usage_available=False,
        )
    )
    artifact = EvaluationTelemetryArtifact.from_scoring(
        score_bundle=bundle,
        score_context=_unavailable_context(
            static_graph_available=True,
            resolved_prediction_count=0,
            resolved_gold_count=1,
            issue_anchor_count=1,
            token_usage_available=False,
        ),
    )

    assert "gold_hop" not in artifact.metric_map
    assert artifact.canonical_component_states["gold_hop"].available is False
    assert artifact.canonical_component_states["gold_hop"].reason == "prediction nodes unresolved"
    assert artifact.canonical_component_states["issue_hop"].reason == "prediction nodes unresolved"
    assert artifact.canonical_component_states["token_efficiency"].reason == "previous_total_token_count unavailable"


def test_localization_task_emitter_scores_and_metadata(monkeypatch: pytest.MonkeyPatch):
    scores: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []

    class Span:
        id = "localization-task"
        name = "localization_task"

        def update(self, **kwargs: object) -> None:
            updates.append(kwargs)

        def score(self, **kwargs: object) -> None:
            scores.append(dict(kwargs))
    artifact = EvaluationTelemetryArtifact.from_scoring(
        score_bundle=_bundle(),
        score_context=_context(),
    )

    evaluation_emitters.emit_localization_task_telemetry(
        Span(),
        artifact,
        score_prefix="ic",
        telemetry_payload={"identity": {"task_id": "task"}},
        predicted_files=["a.py", "missing.py"],
        changed_files=["a.py"],
    )

    assert [score["name"] for score in scores] == [
        "ic.composed_score",
        "ic.composed_score_available",
        "ic.gold_hop",
        "ic.gold_hop_available",
        "ic.issue_hop",
        "ic.issue_hop_available",
        "ic.token_efficiency",
        "ic.token_efficiency_available",
    ]
    metadata = updates[-1]["metadata"]
    assert metadata["score_context"]["gold_min_hops"] == 0
    assert metadata["score_context"]["previous_total_token_count"] == 123.0
    assert metadata["predicted_files"] == ["a.py", "missing.py"]
    assert metadata["score_components"]["gold_hop"] == {
        "available": True,
        "value": 1.0,
        "reason": None,
    }


def test_localization_task_emits_availability_and_metadata_for_unavailable_scores(
    monkeypatch: pytest.MonkeyPatch,
):
    scores: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []

    class Span:
        id = "localization-task"
        name = "localization_task"

        def update(self, **kwargs: object) -> None:
            updates.append(kwargs)

        def score(self, **kwargs: object) -> None:
            scores.append(dict(kwargs))
    unavailable_context = _unavailable_context(
        static_graph_available=False,
        resolved_prediction_count=0,
        resolved_gold_count=0,
        issue_anchor_count=0,
        token_usage_available=False,
    )
    artifact = EvaluationTelemetryArtifact.from_scoring(
        score_bundle=ScoreEngine().evaluate(unavailable_context),
        score_context=unavailable_context,
    )

    evaluation_emitters.emit_localization_task_telemetry(
        Span(),
        artifact,
        score_prefix="ic",
        telemetry_payload={"identity": {"task_id": "task"}},
        predicted_files=["a.py"],
        changed_files=["b.py"],
    )

    score_names = [score["name"] for score in scores]
    assert "ic.gold_hop" not in score_names
    assert "ic.issue_hop" not in score_names
    assert "ic.token_efficiency" not in score_names
    assert "ic.gold_hop_available" in score_names
    assert "ic.issue_hop_available" in score_names
    assert "ic.token_efficiency_available" in score_names
    metadata = updates[-1]["metadata"]["score_components"]
    assert metadata["gold_hop"] == {
        "available": False,
        "value": None,
        "reason": "static_graph unavailable",
    }
    assert metadata["issue_hop"] == {
        "available": False,
        "value": None,
        "reason": "static_graph unavailable",
    }
    assert metadata["token_efficiency"] == {
        "available": False,
        "value": None,
        "reason": "previous_total_token_count unavailable",
    }


def test_optimize_iteration_emits_all_numeric_boolean_metrics(monkeypatch: pytest.MonkeyPatch):
    scores: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []

    class Span:
        id = "optimize-iteration"
        name = "optimize_iteration"

        def update(self, **kwargs: object) -> None:
            updates.append(kwargs)

        def score(self, **kwargs: object) -> None:
            scores.append(dict(kwargs))
    artifact = EvaluationTelemetryArtifact.from_scoring(
        score_bundle=_bundle(),
        score_context=_context(),
    )
    metrics = EvaluationMetrics.model_validate(
        {
            "score": 0.4,
            "additional_metrics": [
                {"name": "gold_hop", "value": 1.0},
                {"name": "issue_hop", "value": 0.5},
                {"name": "token_efficiency", "value": 0.25},
                {"name": "component_passed", "value": True},
                {"name": "ignored", "value": None},
            ],
        }
    )

    evaluation_emitters.emit_optimize_iteration_telemetry(
        Span(),
        metrics=metrics,
        score_component_states=artifact.canonical_component_states,
        iteration=2,
        status="success",
        pipeline_passed=False,
        score_context_summary={"gold_hop": 1.0},
    )

    emitted_names = [score["name"] for score in scores]
    assert emitted_names == [
        "metrics.score",
        "metrics.score_available",
        "gold_hop",
        "gold_hop_available",
        "issue_hop",
        "issue_hop_available",
        "token_efficiency",
        "token_efficiency_available",
        "pipeline_passed",
        "pipeline_passed_available",
        "component_passed",
        "component_passed_available",
    ]
    assert (
        {score["name"]: score["data_type"] for score in scores}["component_passed"]
        == evaluation_emitters.ScoreDataType.BOOLEAN
    )
    assert updates[-1]["metadata"]["score_context_summary"] == {"gold_hop": 1.0}
    assert updates[-1]["metadata"]["score_components"]["gold_hop"]["available"] is True


def test_optimize_iteration_surfaces_unavailable_canonical_component_state(
    monkeypatch: pytest.MonkeyPatch,
):
    scores: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []

    class Span:
        id = "optimize-iteration"
        name = "optimize_iteration"

        def update(self, **kwargs: object) -> None:
            updates.append(kwargs)

        def score(self, **kwargs: object) -> None:
            scores.append(dict(kwargs))
    unavailable_context = _unavailable_context(
        static_graph_available=True,
        resolved_prediction_count=1,
        resolved_gold_count=1,
        issue_anchor_count=0,
        token_usage_available=False,
    )
    artifact = EvaluationTelemetryArtifact.from_scoring(
        score_bundle=ScoreEngine().evaluate(unavailable_context),
        score_context=unavailable_context,
    )
    metrics = EvaluationMetrics.model_validate(
        {
            "score": 0.4,
            "additional_metrics": [
                {"name": "component_passed", "value": True},
            ],
        }
    )

    evaluation_emitters.emit_optimize_iteration_telemetry(
        Span(),
        metrics=metrics,
        score_component_states=artifact.canonical_component_states,
        iteration=1,
        status="failed",
        pipeline_passed=None,
        score_context_summary={},
        error="pipeline unavailable",
    )

    score_names = [score["name"] for score in scores]
    assert "gold_hop_available" in score_names
    assert "issue_hop_available" in score_names
    assert "token_efficiency_available" in score_names
    assert "pipeline_passed_available" in score_names
    assert "issue_hop" not in score_names
    metadata = updates[-1]["metadata"]["score_components"]
    assert metadata["issue_hop"] == {
        "available": False,
        "value": None,
        "reason": "issue anchors unresolved",
    }
    assert metadata["pipeline_passed"] == {
        "available": False,
        "value": None,
        "reason": "pipeline result unavailable",
    }


def test_evaluation_metrics_map_ignores_non_numeric_or_boolean_entries():
    metrics = EvaluationMetrics.model_validate(
        {
            "score": 1.0,
            "additional_metrics": [
                {"name": "numeric", "value": 2},
                {"name": "boolean", "value": False},
                {"name": "missing", "value": None},
            ],
        }
    )

    assert evaluation_metrics_map(metrics) == {
        "metrics.score": 1.0,
        "numeric": 2.0,
        "boolean": False,
    }


def test_unavailable_and_omitted_components_are_distinguishable():
    artifact = EvaluationTelemetryArtifact.from_scoring(
        score_bundle=ScoreEngine().evaluate(
            _unavailable_context(
                static_graph_available=True,
                resolved_prediction_count=0,
                resolved_gold_count=1,
                issue_anchor_count=1,
                token_usage_available=False,
            )
        ),
        score_context=_unavailable_context(
            static_graph_available=True,
            resolved_prediction_count=0,
            resolved_gold_count=1,
            issue_anchor_count=1,
            token_usage_available=False,
        ),
    )

    assert "gold_hop" not in artifact.metric_map
    assert artifact.canonical_component_states["gold_hop"].available is False
    assert "non_canonical_metric" not in artifact.canonical_component_states


def test_metadata_update_failures_are_logged(caplog: pytest.LogCaptureFixture):
    class BrokenSpan:
        id = "broken"

        def update(self, **kwargs: object) -> None:
            raise RuntimeError("metadata down")

    caplog.set_level("WARNING")
    evaluation_emitters._update_observation_metadata(
        BrokenSpan(),
        {"score_context": {"gold_min_hops": 0}},
        operation="test.metadata",
        observation_name="localization_task",
    )

    assert "test.metadata" in caplog.text
    assert "localization_task" in caplog.text
    assert "score_context" in caplog.text
    assert "metadata down" in caplog.text


def test_score_emission_failures_are_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    caplog.set_level("WARNING")

    class Span:
        id = "score-span"
        name = "localization_task"

        def score(self, **kwargs: object) -> None:
            raise RuntimeError("score down")

    evaluation_emitters._emit_score_for_handle(
        Span(),
        name="gold_hop",
        value=1.0,
        data_type=evaluation_emitters.ScoreDataType.NUMERIC,
        operation="test.score",
    )

    assert "test.score" in caplog.text
    assert "gold_hop" in caplog.text
    assert "score down" in caplog.text
