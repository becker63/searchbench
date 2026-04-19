from __future__ import annotations

from typing import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.telemetry.observability_models import TokenUsageBreakdown


class TokenUsage(BaseModel):
    """Raw token counts for a single request."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: float | None = None
    completion_tokens: float | None = None
    total_tokens: float | None = None
    fresh_input_tokens: float | None = None
    cached_input_tokens: float | None = None
    total_input_tokens: float | None = None
    output_tokens: float | None = None


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
    prompt_raw = prompt_raw if isinstance(prompt_raw, (int, float)) else data.get("fresh_input_tokens")
    completion_raw = completion_raw if isinstance(completion_raw, (int, float)) else data.get("output")
    completion_raw = completion_raw if isinstance(completion_raw, (int, float)) else data.get("output_tokens")
    total_raw = total_raw if isinstance(total_raw, (int, float)) else data.get("total")
    cached_raw = data.get("cached_input_tokens")
    cached_raw = cached_raw if isinstance(cached_raw, (int, float)) else data.get("input_cached_tokens")
    total_input_raw = data.get("total_input_tokens")
    if not isinstance(total_input_raw, (int, float)) and any(
        isinstance(value, (int, float)) for value in (prompt_raw, cached_raw)
    ):
        total_input_raw = (float(prompt_raw) if isinstance(prompt_raw, (int, float)) else 0.0) + (
            float(cached_raw) if isinstance(cached_raw, (int, float)) else 0.0
        )
    if not isinstance(total_raw, (int, float)) and isinstance(total_input_raw, (int, float)):
        total_raw = float(total_input_raw) + (
            float(completion_raw) if isinstance(completion_raw, (int, float)) else 0.0
        )

    has_numeric = any(
        isinstance(val, (int, float))
        for val in (prompt_raw, cached_raw, total_input_raw, completion_raw, total_raw)
    )
    raw_model = data.get("model")
    model_name: str | None = raw_model if isinstance(raw_model, str) else None
    usage = TokenUsage(
        prompt_tokens=float(prompt_raw) if isinstance(prompt_raw, (int, float)) else None,
        completion_tokens=float(completion_raw) if isinstance(completion_raw, (int, float)) else None,
        total_tokens=float(total_raw) if isinstance(total_raw, (int, float)) else None,
        fresh_input_tokens=float(prompt_raw) if isinstance(prompt_raw, (int, float)) else None,
        cached_input_tokens=float(cached_raw) if isinstance(cached_raw, (int, float)) else None,
        total_input_tokens=float(total_input_raw) if isinstance(total_input_raw, (int, float)) else None,
        output_tokens=float(completion_raw) if isinstance(completion_raw, (int, float)) else None,
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


def _breakdown_available(tokens: TokenUsageBreakdown) -> bool:
    return any(
        value > 0
        for value in (
            tokens.fresh_input_tokens,
            tokens.cached_input_tokens,
            tokens.total_input_tokens,
            tokens.output_tokens,
            tokens.total_tokens,
        )
    )


def _record_from_breakdown(
    tokens: TokenUsageBreakdown,
    *,
    source: str,
    details: Mapping[str, object] | None = None,
) -> TokenUsageRecord:
    if not _breakdown_available(tokens):
        return TokenUsageRecord(available=False, usage=None, source=source)
    return TokenUsageRecord(
        available=True,
        source=source,
        usage=TokenUsage(
            prompt_tokens=tokens.fresh_input_tokens,
            completion_tokens=tokens.output_tokens,
            total_tokens=tokens.total_tokens,
            fresh_input_tokens=tokens.fresh_input_tokens,
            cached_input_tokens=tokens.cached_input_tokens,
            total_input_tokens=tokens.total_input_tokens,
            output_tokens=tokens.output_tokens,
        ),
        details=details
        or {
            "fresh_input_tokens": tokens.fresh_input_tokens,
            "cached_input_tokens": tokens.cached_input_tokens,
            "total_input_tokens": tokens.total_input_tokens,
            "output_tokens": tokens.output_tokens,
            "total_tokens": tokens.total_tokens,
        },
    )


def _breakdown_from_usage_mapping(data: Mapping[str, object] | None) -> TokenUsageBreakdown:
    return TokenUsageBreakdown.from_usage_details(data if isinstance(data, Mapping) else None)


def _direct_breakdown(container: Mapping[str, object]) -> TokenUsageBreakdown:
    usage_details = container.get("usage_details")
    if isinstance(usage_details, Mapping):
        return _breakdown_from_usage_mapping(usage_details)
    usage = container.get("usage")
    if isinstance(usage, Mapping):
        return _breakdown_from_usage_mapping(usage)
    return TokenUsageBreakdown()


def _observations_breakdown(container: Mapping[str, object]) -> TokenUsageBreakdown:
    observations = container.get("observations")
    if not isinstance(observations, list):
        return TokenUsageBreakdown()
    entries: list[TokenUsageBreakdown] = []
    for obs in observations:
        if not isinstance(obs, Mapping):
            continue
        direct = _direct_breakdown(obs)
        if _breakdown_available(direct):
            entries.append(direct)
    return TokenUsageBreakdown.sum(entries)


def _container_breakdown(container: Mapping[str, object]) -> TokenUsageBreakdown:
    direct = _direct_breakdown(container)
    if _breakdown_available(direct):
        # In agent raw payloads, direct usage_details is already the aggregate
        # over observation-level generation usage. Prefer it to avoid counting
        # mirrored observations twice.
        return direct
    return _observations_breakdown(container)


def _summary_breakdown(payload: Mapping[str, object]) -> TokenUsageBreakdown:
    for key in ("observability_summary", "task_summary"):
        summary = payload.get(key)
        if not isinstance(summary, Mapping):
            continue
        tokens = summary.get("tokens")
        if isinstance(tokens, Mapping):
            breakdown = TokenUsageBreakdown.model_validate(tokens)
            if _breakdown_available(breakdown):
                return breakdown
    return TokenUsageBreakdown()


def aggregate_token_usage_record(
    payload: Mapping[str, object] | None,
    source: str = "runner",
) -> TokenUsageRecord:
    """
    Aggregate token usage for scoring over the full run.

    `extract_token_usage_record` intentionally keeps first-match semantics for
    legacy callers. Token-efficiency scoring needs the full run cost, so this
    helper prefers the canonical task observability summary when present and
    otherwise sums the distinct runner surfaces that represent agent execution
    plus finalization.
    """
    if not isinstance(payload, Mapping):
        return TokenUsageRecord(available=False, usage=None, source=source)

    summary_tokens = _summary_breakdown(payload)
    if _breakdown_available(summary_tokens):
        return _record_from_breakdown(
            summary_tokens,
            source=source,
            details={"aggregation_source": "observability_summary"},
        )

    raw = payload.get("raw")
    if isinstance(raw, Mapping):
        tokens = TokenUsageBreakdown()
        raw_direct = _direct_breakdown(raw)
        if _breakdown_available(raw_direct):
            tokens += raw_direct
        agent_raw = raw.get("raw")
        if isinstance(agent_raw, Mapping):
            tokens += _container_breakdown(agent_raw)
        fallback_raw = raw.get("fallback_raw")
        if isinstance(fallback_raw, Mapping):
            tokens += _container_breakdown(fallback_raw)
        if _breakdown_available(tokens):
            return _record_from_breakdown(
                tokens,
                source=source,
                details={"aggregation_source": "raw"},
            )

    tokens = _container_breakdown(payload)
    if _breakdown_available(tokens):
        return _record_from_breakdown(
            tokens,
            source=source,
            details={"aggregation_source": "payload"},
        )
    return TokenUsageRecord(available=False, usage=None, source=source)


__all__ = [
    "TokenUsage",
    "TokenUsageRecord",
    "aggregate_token_usage_record",
    "extract_token_usage_record",
]
