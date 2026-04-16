from __future__ import annotations

import pytest
from pydantic import ValidationError

from harness.localization.scoring_models import (
    ComposeMode,
    ScoreConfig,
    ScoreContext,
    ScoreEngine,
    summarize_batch_scores,
    summarize_task_score,
)


def test_score_context_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ScoreContext(gold_min_hops=1, metrics={"score": 1.0})


def test_default_composition_keeps_token_efficiency_best_effort():
    ctx = ScoreContext(
        predicted_files=("src/b.py",),
        gold_files=("src/a.py",),
        gold_min_hops=1,
        issue_min_hops=2,
        per_prediction_hops={"src/b.py": 1},
    )
    bundle = ScoreEngine().evaluate(ctx)

    assert tuple(bundle.results) == ("gold_hop", "issue_hop", "token_efficiency")
    assert bundle.results["token_efficiency"].available is False
    assert bundle.results["token_efficiency"].value is None
    assert bundle.composed_score == pytest.approx((0.5 * 0.75) + ((1.0 / 3.0) * 0.25))
    assert bundle.available is True


def test_token_efficiency_can_be_required_by_composition():
    config = ScoreConfig(
        enabled=("gold_hop", "issue_hop", "token_efficiency"),
        visible_to_agent=("gold_hop", "issue_hop", "token_efficiency"),
        compose_weights={"gold_hop": 0.5, "token_efficiency": 0.5},
    )
    ctx = ScoreContext(gold_min_hops=0, issue_min_hops=1)
    bundle = ScoreEngine(config=config).evaluate(ctx)

    assert bundle.results["token_efficiency"].available is False
    assert bundle.composed_score is None
    assert bundle.available is False


def test_visibility_override_and_geometric_composition():
    config = ScoreConfig(
        enabled=("gold_hop", "issue_hop", "token_efficiency"),
        visible_to_agent=("gold_hop",),
        compose_weights={"gold_hop": 0.5, "issue_hop": 0.5},
        compose_mode=ComposeMode.GEOMETRIC_MEAN,
    )
    ctx = ScoreContext(gold_min_hops=0, issue_min_hops=3, previous_total_token_count=10)
    bundle = ScoreEngine(config=config).evaluate(ctx)

    assert bundle.results["gold_hop"].visible_to_agent is True
    assert bundle.results["issue_hop"].visible_to_agent is False
    assert bundle.results["token_efficiency"].visible_to_agent is False
    assert bundle.composed_score == pytest.approx((1.0**0.5) * (0.25**0.5))


def test_invalid_config_rejects_disabled_visible_or_composed_scores():
    with pytest.raises(ValidationError):
        ScoreConfig(
            enabled=("gold_hop",),
            visible_to_agent=("issue_hop",),
            compose_weights={"gold_hop": 1.0},
        )

    with pytest.raises(ValidationError):
        ScoreConfig(
            enabled=("gold_hop",),
            visible_to_agent=("gold_hop",),
            compose_weights={"issue_hop": 1.0},
        )


def test_unknown_registry_entry_is_rejected():
    config = ScoreConfig(
        enabled=("gold_hop", "missing_score"),
        visible_to_agent=("gold_hop",),
        compose_weights={"gold_hop": 1.0},
    )
    with pytest.raises(KeyError):
        ScoreEngine(config=config)


def test_task_and_batch_summaries_include_counts():
    first = ScoreEngine().evaluate(
        ScoreContext(gold_min_hops=0, issue_min_hops=1, previous_total_token_count=10)
    )
    second = ScoreEngine().evaluate(
        ScoreContext(gold_min_hops=2, issue_min_hops=2)
    )

    task_summary = summarize_task_score("task-1", first)
    assert task_summary.task_id == "task-1"
    assert task_summary.component_availability["token_efficiency"] is True

    batch = summarize_batch_scores({"task-1": first, "task-2": second})
    assert batch.total_count == 2
    assert batch.composed_available_count == 2
    assert batch.components["gold_hop"].available_count == 2
    assert batch.components["token_efficiency"].available_count == 1
    assert batch.components["token_efficiency"].missing_count == 1
