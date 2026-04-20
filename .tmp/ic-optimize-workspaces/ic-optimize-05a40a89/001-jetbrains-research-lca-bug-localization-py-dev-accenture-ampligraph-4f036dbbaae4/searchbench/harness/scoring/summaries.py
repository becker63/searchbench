from __future__ import annotations

from typing import Mapping

from .batch import (
    AggregateComponentScore,
    BatchScoreSummary,
)
from .bundle import ScoreBundle


def summarize_task_score(task_id: str, bundle: ScoreBundle) -> ScoreBundle:
    """Attach a localization task id to a concrete `ScoreBundle`."""

    return bundle.model_copy(update={"item_id": task_id})


def summarize_score_summaries(
    summaries: Mapping[str, ScoreBundle],
) -> BatchScoreSummary:
    """Aggregate task score bundles into the localization batch score summary."""

    total_count = len(summaries)
    composed_values = [
        summary.composed_score
        for summary in summaries.values()
        if summary.composed_score is not None
    ]
    component_names = sorted(
        {name for summary in summaries.values() for name in summary.component_scores}
    )
    components: dict[str, AggregateComponentScore] = {}
    for name in component_names:
        values: list[float] = []
        visible = False
        for summary in summaries.values():
            visible = visible or name in summary.visible_component_scores
            value = summary.component_scores.get(name)
            if value is not None:
                values.append(value)
        available_count = len(values)
        components[name] = AggregateComponentScore(
            name=name,
            average=(sum(values) / available_count) if values else None,
            available_count=available_count,
            missing_count=total_count - available_count,
            total_count=total_count,
            visible_to_agent=visible,
        )
    composed_available_count = len(composed_values)
    return BatchScoreSummary(
        aggregate_composed_score=(sum(composed_values) / composed_available_count)
        if composed_values
        else None,
        composed_available_count=composed_available_count,
        composed_missing_count=total_count - composed_available_count,
        total_count=total_count,
        items=dict(summaries),
        components=components,
    )


def summarize_batch_scores(bundles: Mapping[str, ScoreBundle]) -> BatchScoreSummary:
    """
    Project concrete bundles into item summaries, then aggregate those summaries.
    """

    summaries = {
        task_id: summarize_task_score(task_id, bundle)
        for task_id, bundle in bundles.items()
    }
    return summarize_score_summaries(summaries)


__all__ = [
    "summarize_batch_scores",
    "summarize_score_summaries",
    "summarize_task_score",
]
