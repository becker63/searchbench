from __future__ import annotations

from typing import Generic, Mapping, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .bundle import ScoreBundle

TItemBundle = TypeVar("TItemBundle", bound=ScoreBundle)


class ComponentAggregate(BaseModel):
    """Aggregate batch-level summary for one named score component."""

    model_config = ConfigDict(extra="forbid")

    name: str
    average: float | None
    available_count: int
    missing_count: int
    total_count: int
    visible_to_agent: bool


class BatchSummary(BaseModel, Generic[TItemBundle]):
    """Generic aggregate summary of score outcomes across a batch of items."""

    model_config = ConfigDict(extra="forbid")

    aggregate_composed_score: float | None
    composed_available_count: int
    composed_missing_count: int
    total_count: int
    items: Mapping[str, TItemBundle] = Field(default_factory=dict)
    components: Mapping[str, ComponentAggregate] = Field(default_factory=dict)


TaskScoreSummary: TypeAlias = ScoreBundle
AggregateComponentScore: TypeAlias = ComponentAggregate
BatchScoreSummary: TypeAlias = BatchSummary[TaskScoreSummary]


__all__ = [
    "AggregateComponentScore",
    "BatchScoreSummary",
    "BatchSummary",
    "ComponentAggregate",
    "TaskScoreSummary",
]
