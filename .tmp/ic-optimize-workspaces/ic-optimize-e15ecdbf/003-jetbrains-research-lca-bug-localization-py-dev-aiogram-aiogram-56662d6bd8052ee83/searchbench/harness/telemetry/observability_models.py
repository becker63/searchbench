from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


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


__all__ = [
    "BackendRunStats",
    "RunSummary",
    "TaskRunSummary",
    "TokenUsageBreakdown",
]
