from __future__ import annotations

from enum import Enum
from typing import Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Goal(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ComposeMode(str, Enum):
    WEIGHTED_SUM = "weighted_sum"
    GEOMETRIC_MEAN = "geometric_mean"


class ScoreContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_files: tuple[str, ...] = Field(default_factory=tuple)
    gold_files: tuple[str, ...] = Field(default_factory=tuple)
    gold_min_hops: int | None = None
    issue_min_hops: int | None = None
    per_prediction_hops: Mapping[str, int | None] = Field(default_factory=dict)
    previous_total_token_count: float | None = None
    resolved_prediction_nodes: Mapping[str, tuple[str, ...]] = Field(
        default_factory=dict
    )
    resolved_gold_nodes: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)
    issue_anchor_nodes: tuple[str, ...] = Field(default_factory=tuple)
    unresolved_predictions: tuple[str, ...] = Field(default_factory=tuple)
    unresolved_gold_files: tuple[str, ...] = Field(default_factory=tuple)
    repo_path: str | None = None
    diagnostics: Mapping[str, object] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | None
    goal: Goal
    available: bool
    visible_to_agent: bool
    reason: str | None = None
    metadata: Mapping[str, object] = Field(default_factory=dict)


class ScoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: tuple[str, ...]
    visible_to_agent: tuple[str, ...]
    compose_weights: Mapping[str, float]
    compose_mode: ComposeMode = ComposeMode.WEIGHTED_SUM
    epsilon: float = 1e-6

    @model_validator(mode="after")
    def _validate_config(self) -> "ScoreConfig":
        enabled = set(self.enabled)
        visible = set(self.visible_to_agent)
        composed = set(self.compose_weights)
        if not self.enabled:
            raise ValueError("enabled must not be empty")
        if not self.compose_weights:
            raise ValueError("compose_weights must not be empty")
        if not visible <= enabled:
            raise ValueError(
                f"Visible scores must also be enabled: {sorted(visible - enabled)}"
            )
        if not composed <= enabled:
            raise ValueError(
                f"Composed scores must also be enabled: {sorted(composed - enabled)}"
            )
        total_weight = sum(float(weight) for weight in self.compose_weights.values())
        if total_weight <= 0.0:
            raise ValueError("compose_weights must sum to > 0")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        return self

    @classmethod
    def default(cls) -> "ScoreConfig":
        return cls(
            enabled=("gold_hop", "issue_hop", "token_efficiency"),
            visible_to_agent=("gold_hop", "issue_hop", "token_efficiency"),
            compose_weights={
                "gold_hop": 0.75,
                "issue_hop": 0.25,
            },
            compose_mode=ComposeMode.WEIGHTED_SUM,
        )


class ScoreBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: Mapping[str, ScoreResult]
    composed_score: float | None
    compose_mode: ComposeMode
    compose_weights: Mapping[str, float]
    available: bool
    diagnostics: Mapping[str, object] = Field(default_factory=dict)


class TaskScoreSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    composed_score: float | None
    available: bool
    component_scores: Mapping[str, float | None] = Field(default_factory=dict)
    visible_component_scores: Mapping[str, float | None] = Field(default_factory=dict)
    component_availability: Mapping[str, bool] = Field(default_factory=dict)


class AggregateComponentScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    average: float | None
    available_count: int
    missing_count: int
    total_count: int
    visible_to_agent: bool


class BatchScoreSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregate_composed_score: float | None
    composed_available_count: int
    composed_missing_count: int
    total_count: int
    components: Mapping[str, AggregateComponentScore] = Field(default_factory=dict)


class ScoreComponent(Protocol):
    name: str
    default_enabled: bool
    default_visible_to_agent: bool

    def compute(self, ctx: ScoreContext) -> ScoreResult: ...


def summarize_task_score(task_id: str, bundle: ScoreBundle) -> TaskScoreSummary:
    component_scores = {name: result.value for name, result in bundle.results.items()}
    visible_scores = {
        name: result.value
        for name, result in bundle.results.items()
        if result.visible_to_agent
    }
    availability = {name: result.available for name, result in bundle.results.items()}
    return TaskScoreSummary(
        task_id=task_id,
        composed_score=bundle.composed_score,
        available=bundle.available,
        component_scores=component_scores,
        visible_component_scores=visible_scores,
        component_availability=availability,
    )


def summarize_batch_scores(bundles: Mapping[str, ScoreBundle]) -> BatchScoreSummary:
    total_count = len(bundles)
    composed_values = [
        bundle.composed_score
        for bundle in bundles.values()
        if bundle.composed_score is not None
    ]
    component_names = sorted(
        {name for bundle in bundles.values() for name in bundle.results}
    )
    components: dict[str, AggregateComponentScore] = {}
    for name in component_names:
        values: list[float] = []
        visible = False
        for bundle in bundles.values():
            result = bundle.results.get(name)
            if result is None:
                continue
            visible = visible or result.visible_to_agent
            if result.available and result.value is not None:
                values.append(result.value)
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
        components=components,
    )


__all__ = [
    "AggregateComponentScore",
    "BatchScoreSummary",
    "ComposeMode",
    "Goal",
    "ScoreBundle",
    "ScoreComponent",
    "ScoreConfig",
    "ScoreContext",
    "ScoreResult",
    "TaskScoreSummary",
    "summarize_batch_scores",
    "summarize_task_score",
]
