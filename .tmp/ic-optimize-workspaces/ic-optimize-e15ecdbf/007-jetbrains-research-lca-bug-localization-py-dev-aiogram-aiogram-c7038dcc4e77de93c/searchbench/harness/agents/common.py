"""
Shared agent utilities that are reused by runner and writer agents.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harness.localization.models import LCATask
from harness.telemetry.observability_models import TokenUsageBreakdown


class UsageEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float


class UsageEnvelope(BaseModel):
    """Canonical usage details envelope aligned with observability expectations."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: float | None = None
    completion_tokens: float | None = None
    total_tokens: float | None = None
    input: float | None = None
    output: float | None = None
    total: float | None = None
    prompt_details: list[UsageEntry] = Field(default_factory=list)
    completion_details: list[UsageEntry] = Field(default_factory=list)

    def model_dump_flat(self) -> dict[str, float]:
        data: dict[str, float] = {}
        if self.prompt_tokens is not None:
            data["prompt_tokens"] = float(self.prompt_tokens)
        if self.completion_tokens is not None:
            data["completion_tokens"] = float(self.completion_tokens)
        if self.total_tokens is not None:
            data["total_tokens"] = float(self.total_tokens)
        if self.input is not None:
            data["input"] = float(self.input)
        elif self.prompt_tokens is not None:
            data["input"] = float(self.prompt_tokens)
        if self.output is not None:
            data["output"] = float(self.output)
        elif self.completion_tokens is not None:
            data["output"] = float(self.completion_tokens)
        if self.total is not None:
            data["total"] = float(self.total)
        elif self.total_tokens is not None:
            data["total"] = float(self.total_tokens)
        for entry in self.prompt_details:
            data[f"prompt_tokens_details.{entry.name}"] = float(entry.value)
            data[f"input_{entry.name}"] = float(entry.value)
        for entry in self.completion_details:
            data[f"completion_tokens_details.{entry.name}"] = float(entry.value)
            data[f"output_{entry.name}"] = float(entry.value)
        return data


def usage_from_response(resp: Any) -> UsageEnvelope | None:
    """
    Normalize OpenAI-style usage into Langfuse usage_details shape.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_entries: list[UsageEntry] = []
    completion_entries: list[UsageEntry] = []
    if isinstance(prompt_details, Mapping):
        for key, val in prompt_details.items():
            if isinstance(val, (int, float)):
                prompt_entries.append(UsageEntry(name=str(key), value=float(val)))
    if isinstance(completion_details, Mapping):
        for key, val in completion_details.items():
            if isinstance(val, (int, float)):
                completion_entries.append(UsageEntry(name=str(key), value=float(val)))
    return UsageEnvelope(
        prompt_tokens=float(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None,
        completion_tokens=float(completion_tokens) if isinstance(completion_tokens, (int, float)) else None,
        total_tokens=float(total_tokens) if isinstance(total_tokens, (int, float)) else None,
        input=float(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None,
        output=float(completion_tokens) if isinstance(completion_tokens, (int, float)) else None,
        total=float(total_tokens) if isinstance(total_tokens, (int, float)) else None,
        prompt_details=prompt_entries,
        completion_details=completion_entries,
    )


def normalize_token_usage(
    usage_details: Mapping[str, object] | None,
) -> TokenUsageBreakdown:
    """Normalize Langfuse/provider usage_details into the shared observability shape."""
    return TokenUsageBreakdown.from_usage_details(usage_details)


def serialize_lca_task_for_prompt(task: LCATask) -> dict[str, object]:
    """
    Convert the canonical internal task into the localizer prompt transport payload.
    Gold labels stay out of the prompt payload.
    """
    if not task.repo:
        raise ValueError("LCATask.repo is required for localizer prompt serialization")
    return {
        "identity": task.identity.model_dump(),
        "repo": task.repo,
        "context": task.context.model_dump(),
    }


__all__ = ["normalize_token_usage", "serialize_lca_task_for_prompt", "usage_from_response"]
