from __future__ import annotations

from typing import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TokenUsage(BaseModel):
    """Raw token counts for a single request."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: float | None = None
    completion_tokens: float | None = None
    total_tokens: float | None = None


class TokenUsageRecord(BaseModel):
    """Token usage plus availability metadata to distinguish missing from zero."""

    model_config = ConfigDict(extra="forbid")

    available: bool = False
    usage: TokenUsage | None = None
    model: str | None = None
    source: str | None = None
    details: Mapping[str, object] | None = Field(default=None, description="Raw provider usage payload if available")

    @model_validator(mode="after")
    def _normalize_usage(self) -> "TokenUsageRecord":
        if not self.available:
            self.usage = None
        return self

    def normalized_total(self) -> float | None:
        """Prefer explicit total_tokens; otherwise sum prompt+completion when available."""
        if not self.available or self.usage is None:
            return None
        if self.usage.total_tokens is not None:
            return float(self.usage.total_tokens)
        prompt = self.usage.prompt_tokens
        completion = self.usage.completion_tokens
        if prompt is None and completion is None:
            return None
        return float((prompt or 0.0) + (completion or 0.0))


def _usage_from_mapping(data: Mapping[str, object]) -> TokenUsageRecord:
    prompt_raw = data.get("prompt_tokens") if isinstance(data, Mapping) else None
    completion_raw = data.get("completion_tokens") if isinstance(data, Mapping) else None
    total_raw = data.get("total_tokens") if isinstance(data, Mapping) else None
    # Support alternate keys used in usage payloads.
    prompt_raw = prompt_raw if isinstance(prompt_raw, (int, float)) else data.get("input")
    completion_raw = completion_raw if isinstance(completion_raw, (int, float)) else data.get("output")
    total_raw = total_raw if isinstance(total_raw, (int, float)) else data.get("total")

    has_numeric = any(isinstance(val, (int, float)) for val in (prompt_raw, completion_raw, total_raw))
    model_name = data.get("model") if isinstance(data.get("model"), str) else None
    usage = TokenUsage(
        prompt_tokens=float(prompt_raw) if isinstance(prompt_raw, (int, float)) else None,
        completion_tokens=float(completion_raw) if isinstance(completion_raw, (int, float)) else None,
        total_tokens=float(total_raw) if isinstance(total_raw, (int, float)) else None,
    )
    return TokenUsageRecord(
        available=has_numeric,
        usage=usage if has_numeric else None,
        model=model_name,
        details=data,
    )


def _iter_usage_mappings(payload: Mapping[str, object] | None) -> Iterable[Mapping[str, object]]:
    if not isinstance(payload, Mapping):
        return
    # Priority: top-level usage_details, then usage, then raw block, then observations.
    direct = payload.get("usage_details")
    if isinstance(direct, Mapping):
        yield direct
    direct_usage = payload.get("usage")
    if isinstance(direct_usage, Mapping):
        yield direct_usage
    raw_block = payload.get("raw")
    if isinstance(raw_block, Mapping):
        raw_usage_details = raw_block.get("usage_details")
        if isinstance(raw_usage_details, Mapping):
            yield raw_usage_details
        raw_usage = raw_block.get("usage")
        if isinstance(raw_usage, Mapping):
            yield raw_usage
    observations = payload.get("observations")
    if isinstance(observations, list):
        for obs in observations:
            if not isinstance(obs, Mapping):
                continue
            obs_usage_details = obs.get("usage_details")
            if isinstance(obs_usage_details, Mapping):
                yield obs_usage_details
            obs_usage = obs.get("usage")
            if isinstance(obs_usage, Mapping):
                yield obs_usage
    return


def extract_token_usage_record(payload: Mapping[str, object] | None, source: str = "runner") -> TokenUsageRecord:
    """
    Extract token usage from known shapes without fabricating values.
    Supported precedence:
    - top-level usage_details
    - top-level usage
    - raw.usage_details / raw.usage
    - observations[*].usage_details / observations[*].usage
    Missing usage remains available=False with null fields.
    """
    for usage_map in _iter_usage_mappings(payload):
        record = _usage_from_mapping(usage_map)
        if record.available:
            record.source = source
            return record
    return TokenUsageRecord(available=False, usage=None, source=source)


__all__ = ["TokenUsage", "TokenUsageRecord", "extract_token_usage_record"]
