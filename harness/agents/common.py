from __future__ import annotations

"""
Shared agent utilities that are reused by runner and writer agents.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


__all__ = ["usage_from_response"]


class AgentTaskPayload(BaseModel):
    """Validated agent task payload for runner agents."""

    model_config = ConfigDict(extra="forbid")

    identity: dict[str, object]
    repo: str
    context: dict[str, object]
    extra: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_blocks(cls, value: object) -> object:
        if isinstance(value, Mapping):
            identity_val = value.get("identity")
            context_val = value.get("context")

            def _maybe_dump(obj: object) -> object:
                model_dump = getattr(obj, "model_dump", None)
                if callable(model_dump):
                    try:
                        return model_dump()
                    except Exception:
                        return obj
                return obj

            normalized_identity = _maybe_dump(identity_val)
            normalized_context = _maybe_dump(context_val)
            return {
                **value,
                "identity": normalized_identity,
                "context": normalized_context,
            }
        return value
