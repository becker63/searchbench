from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from harness.scoring import ScoreBundle, ScoreContext


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _usage_value(usage: Mapping[str, object], *keys: str) -> float:
    for key in keys:
        value = _number(usage.get(key))
        if value is not None:
            return value
    return 0.0


class TokenUsageBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fresh_input_tokens: float = 0.0
    cached_input_tokens: float = 0.0
    total_input_tokens: float = 0.0
    output_tokens: float = 0.0
    total_tokens: float = 0.0

    @classmethod
    def from_usage_details(
        cls, usage_details: Mapping[str, object] | None
    ) -> "TokenUsageBreakdown":
        if not isinstance(usage_details, Mapping):
            return cls()
        fresh_input = _usage_value(usage_details, "input", "prompt_tokens")
        cached_input = _usage_value(
            usage_details,
            "input_cached_tokens",
            "input_cache_read",
            "prompt_tokens_details.cached_tokens",
        )
        output = _usage_value(usage_details, "output", "completion_tokens")
        explicit_total = _number(usage_details.get("total"))
        if explicit_total is None:
            explicit_total = _number(usage_details.get("total_tokens"))
        total_input = fresh_input + cached_input
        return cls(
            fresh_input_tokens=fresh_input,
            cached_input_tokens=cached_input,
            total_input_tokens=total_input,
            output_tokens=output,
            total_tokens=explicit_total if explicit_total is not None else total_input + output,
        )

    @classmethod
    def sum(cls, entries: Iterable["TokenUsageBreakdown"]) -> "TokenUsageBreakdown":
        total = cls()
        for entry in entries:
            total = total + entry
        return total

    def __add__(self, other: "TokenUsageBreakdown") -> "TokenUsageBreakdown":
        return TokenUsageBreakdown(
            fresh_input_tokens=self.fresh_input_tokens + other.fresh_input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            total_input_tokens=self.total_input_tokens + other.total_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class BackendRunStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_count: int = 0
    generation_count: int = 0
    tool_calls_count: int = 0
    successful_tool_calls_count: int = 0
    failed_tool_calls_count: int = 0


class TaskRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["ic", "jc"]
    identity: str
    repo: str
    status: Literal["success", "failure"]
    stage: str | None = None
    message: str | None = None
    step_count: int = 0
    generation_count: int = 0
    tool_calls_count: int = 0
    successful_tool_calls_count: int = 0
    failed_tool_calls_count: int = 0
    predicted_files_count: int = 0
    duration_seconds: float = 0.0
    tokens: TokenUsageBreakdown = Field(default_factory=TokenUsageBreakdown)
    evaluation: dict[str, float] = Field(default_factory=dict)


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["ic", "jc"]
    selected_tasks_count: int
    completed_tasks_count: int
    successful_tasks_count: int
    failed_tasks_count: int
    tokens: TokenUsageBreakdown = Field(default_factory=TokenUsageBreakdown)
    average_tokens_per_task: TokenUsageBreakdown = Field(default_factory=TokenUsageBreakdown)
    total_tool_calls_count: int = 0
    total_successful_tool_calls_count: int = 0
    total_failed_tool_calls_count: int = 0
    average_tool_calls_per_task: float = 0.0
    total_generations_count: int = 0
    average_generations_per_task: float = 0.0
    total_duration_seconds: float = 0.0
    average_task_duration_seconds: float = 0.0
    failure_stages: dict[str, int] = Field(default_factory=dict)

    @computed_field
    @property
    def success_rate(self) -> float:
        if self.completed_tasks_count <= 0:
            return 0.0
        return self.successful_tasks_count / self.completed_tasks_count

    @classmethod
    def from_task_summaries(
        cls,
        *,
        backend: Literal["ic", "jc"],
        selected_tasks_count: int,
        task_summaries: Iterable[TaskRunSummary],
    ) -> "RunSummary":
        summaries = list(task_summaries)
        completed = len(summaries)
        successful = sum(1 for summary in summaries if summary.status == "success")
        failed = sum(1 for summary in summaries if summary.status == "failure")
        tokens = TokenUsageBreakdown.sum(summary.tokens for summary in summaries)
        divisor = completed or 1
        failure_stages: dict[str, int] = {}
        for summary in summaries:
            if summary.status == "failure" and summary.stage:
                failure_stages[summary.stage] = failure_stages.get(summary.stage, 0) + 1
        total_tool_calls = sum(summary.tool_calls_count for summary in summaries)
        total_generations = sum(summary.generation_count for summary in summaries)
        total_duration = sum(summary.duration_seconds for summary in summaries)
        return cls(
            backend=backend,
            selected_tasks_count=selected_tasks_count,
            completed_tasks_count=completed,
            successful_tasks_count=successful,
            failed_tasks_count=failed,
            tokens=tokens,
            average_tokens_per_task=TokenUsageBreakdown(
                fresh_input_tokens=tokens.fresh_input_tokens / divisor,
                cached_input_tokens=tokens.cached_input_tokens / divisor,
                total_input_tokens=tokens.total_input_tokens / divisor,
                output_tokens=tokens.output_tokens / divisor,
                total_tokens=tokens.total_tokens / divisor,
            ),
            total_tool_calls_count=total_tool_calls,
            total_successful_tool_calls_count=sum(
                summary.successful_tool_calls_count for summary in summaries
            ),
            total_failed_tool_calls_count=sum(
                summary.failed_tool_calls_count for summary in summaries
            ),
            average_tool_calls_per_task=total_tool_calls / divisor,
            total_generations_count=total_generations,
            average_generations_per_task=total_generations / divisor,
            total_duration_seconds=total_duration,
            average_task_duration_seconds=total_duration / divisor,
            failure_stages=failure_stages,
        )


class CompactScoreContext(BaseModel):
    """Bounded score diagnostics for observation metadata.

    Langfuse scores carry evaluation values. This context is explanatory
    metadata for the single observation being evaluated, not propagated trace
    metadata and not a raw `ScoreContext` dump.
    """

    model_config = ConfigDict(extra="forbid")

    gold_min_hops: int | None = None
    issue_min_hops: int | None = None
    previous_total_token_count: float | None = None
    static_graph_available: bool | None = None
    token_usage_available: bool | None = None
    gold_min_hops_available: bool | None = None
    issue_min_hops_available: bool | None = None
    predicted_files_count: int = 0
    gold_files_count: int = 0
    resolved_prediction_count: int = 0
    resolved_gold_count: int = 0
    unresolved_prediction_count: int = 0
    unresolved_gold_count: int = 0
    issue_anchor_count: int | None = None
    gold_anchor_count: int | None = None

    @classmethod
    def from_score_context(cls, context: ScoreContext) -> "CompactScoreContext":
        diagnostics = context.diagnostics
        return cls(
            gold_min_hops=context.gold_min_hops,
            issue_min_hops=context.issue_min_hops,
            previous_total_token_count=context.previous_total_token_count,
            static_graph_available=_optional_bool(diagnostics.get("static_graph_available")),
            token_usage_available=_optional_bool(diagnostics.get("token_usage_available")),
            gold_min_hops_available=_optional_bool(diagnostics.get("gold_min_hops_available")),
            issue_min_hops_available=_optional_bool(diagnostics.get("issue_min_hops_available")),
            predicted_files_count=len(context.predicted_files),
            gold_files_count=len(context.gold_files),
            resolved_prediction_count=_int_or_count(
                diagnostics.get("resolved_prediction_count"),
                context.resolved_prediction_nodes,
            ),
            resolved_gold_count=_int_or_count(
                diagnostics.get("resolved_gold_count"),
                context.resolved_gold_nodes,
            ),
            unresolved_prediction_count=len(context.unresolved_predictions),
            unresolved_gold_count=len(context.unresolved_gold_files),
            issue_anchor_count=_optional_int(diagnostics.get("issue_anchor_count")),
            gold_anchor_count=_optional_int(diagnostics.get("gold_anchor_count")),
        )


class LocalizationTaskTelemetryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_context: CompactScoreContext
    predicted_files: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)


class OptimizeIterationTelemetryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int
    iteration_ordinal: int
    status: str | None = None
    pipeline_passed: bool | None = None
    score_context_summary: CompactScoreContext | None = None
    error: str | None = None


class EvaluationTelemetryArtifact(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    score_bundle: ScoreBundle
    score_context: ScoreContext
    score_summary: ScoreBundle
    metric_map: dict[str, float | bool] = Field(default_factory=dict)
    compact_score_context: CompactScoreContext

    @classmethod
    def from_scoring(
        cls,
        *,
        score_bundle: ScoreBundle,
        score_context: ScoreContext,
        score_summary: ScoreBundle | None = None,
    ) -> "EvaluationTelemetryArtifact":
        return cls(
            score_bundle=score_bundle,
            score_context=score_context,
            score_summary=score_summary or score_bundle,
            metric_map=score_bundle_metric_map(score_bundle),
            compact_score_context=CompactScoreContext.from_score_context(score_context),
        )


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _int_or_count(value: object, fallback: Mapping[str, object]) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return sum(1 for item in fallback.values() if item)


def score_bundle_metric_map(bundle: ScoreBundle) -> dict[str, float | bool]:
    metrics: dict[str, float | bool] = {}
    if bundle.composed_score is not None:
        metrics["composed_score"] = float(bundle.composed_score)
    for component_name, result in bundle.results.items():
        stable_name = result.name or component_name
        if not result.visible_to_agent:
            continue
        if not result.available or not isinstance(result.value, (int, float)):
            continue
        metrics[stable_name] = float(result.value)
    return metrics


def evaluation_metrics_map(metrics: object) -> dict[str, float | bool]:
    metric_map: dict[str, float | bool] = {}
    score = getattr(metrics, "score", None)
    if isinstance(score, bool):
        metric_map["metrics.score"] = score
    elif isinstance(score, (int, float)):
        metric_map["metrics.score"] = float(score)
    for entry in getattr(metrics, "additional_metrics", []) or []:
        name = getattr(entry, "name", None)
        value = getattr(entry, "value", None)
        if not isinstance(name, str) or not name:
            continue
        if isinstance(value, bool):
            metric_map[name] = value
        elif isinstance(value, (int, float)):
            metric_map[name] = float(value)
    return metric_map


__all__ = [
    "BackendRunStats",
    "CompactScoreContext",
    "EvaluationTelemetryArtifact",
    "LocalizationTaskTelemetryContext",
    "OptimizeIterationTelemetryContext",
    "RunSummary",
    "TaskRunSummary",
    "TokenUsageBreakdown",
    "evaluation_metrics_map",
    "score_bundle_metric_map",
]
