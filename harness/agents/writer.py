"""
Leaf operation: policy writer/generator.
No CLI or orchestration logic; only generation and validation.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

from langfuse import LangfuseGeneration, LangfuseSpan
from langfuse.api.commons.types import ScoreDataType
from pydantic import BaseModel, ConfigDict, ValidationError

from harness.log import bind_logger, get_logger
from .common import usage_from_response  # pyright: ignore[reportPrivateUsage]
from harness.prompts import (
    WriterOptimizationBrief,
    build_writer_prompt,
)
from harness.telemetry.tracing import (
    get_tracing_openai_client,
    serialize_observation_metadata,
)
from harness.utils.env import get_cerebras_api_key, get_writer_model
from harness.utils.model_budgets import compute_prompt_char_budget, get_model_budget
from harness.utils.type_loader import FrontierContext, format_frontier_context

_LOGGER = get_logger(__name__)


class WriterRequest(BaseModel):
    """Validated boundary for writer requests."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    feedback: Mapping[str, object]
    current_policy: str
    tests: str
    frontier_context: str
    frontier_context_details: FrontierContext
    optimization_brief: WriterOptimizationBrief
    parent_trace: Any | None = None
    writer_session_id: str | None = None
    failure_context: str | None = None
    repair_attempt: int = 0

_client: Any | None = None


def _get_client():
    global _client  # noqa: PLW0603
    if _client is None:
        api_key = get_cerebras_api_key()
        if not api_key:
            raise RuntimeError("Missing CEREBRAS_API_KEY in environment")
        _client = get_tracing_openai_client(base_url="https://api.cerebras.ai/v1", api_key=api_key)
    return _client, get_writer_model()


def _require_parent(
    parent_trace: LangfuseSpan | LangfuseGeneration | None,
) -> LangfuseSpan | LangfuseGeneration:
    if parent_trace is None:
        raise RuntimeError("Parent span required for writer tracing")
    return parent_trace


def _span_label(span: object) -> str:
    span_id = getattr(span, "id", None)
    return span_id if isinstance(span_id, str) and span_id else type(span).__name__


def _langfuse_usage_details(
    usage_details: Mapping[str, object] | None,
) -> dict[str, int] | None:
    if not isinstance(usage_details, Mapping):
        return None
    numeric: dict[str, int] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input",
        "input_cached_tokens",
        "output",
        "total",
    ):
        value = usage_details.get(key)
        if isinstance(value, (int, float)):
            numeric[key] = int(value)
    return numeric or None


def _ensure_non_empty_messages(messages: Sequence[Mapping[str, object]] | None, caller: str) -> None:
    if not messages:
        raise RuntimeError(f"{caller}: empty messages payload")


def _extract_policy_code(response: Any) -> str:
    """
    Extract structured code from a model response.

    Supports:
    - response.choices[0].message.content as JSON string (primary)
    - response.choices[0].message.content as mapping
    """
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Writer response missing choices")
    msg = getattr(choices[0], "message", None)
    if msg is None:
        raise ValueError("Writer response missing message")

    def _mapping_code(obj: Mapping[str, object] | None) -> str | None:
        if obj is None:
            return None
        code_val = obj.get("code")
        return code_val if isinstance(code_val, str) else None

    content = getattr(msg, "content", None)
    if isinstance(content, str):
        try:
            loaded = json.loads(content)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Writer response content is not valid JSON") from exc
        if not isinstance(loaded, Mapping):
            raise ValueError("Writer response content is not a JSON object")
        code_val = _mapping_code(loaded)
        if code_val is not None:
            return code_val
        raise ValueError("Writer response missing code field")
    if isinstance(content, Mapping):
        code_val = _mapping_code(content)
        if code_val is not None:
            return code_val
        raise ValueError("Writer response missing code field")
    if content is not None:
        raise ValueError("Writer response content is not supported")

    raise ValueError("Writer response missing content")


def _ensure_valid_policy_code(code: str) -> str:
    cleaned = code if isinstance(code, str) else ""
    if not cleaned or not cleaned.strip():
        raise ValueError("Writer returned empty code")
    if "```" in cleaned:
        raise ValueError("Writer returned fenced code")
    if "def frontier_priority" not in cleaned:
        raise ValueError("Writer missing frontier_priority function")
    compile(cleaned, "<generated-policy>", "exec")
    return cleaned


def generate_policy(
    current_policy: str,
    tests: str,
    frontier_context: str,
    frontier_context_details: FrontierContext,
    optimization_brief: WriterOptimizationBrief,
    feedback: dict[str, Any] | None = None,
    parent_trace: LangfuseSpan | LangfuseGeneration | None = None,
    writer_session_id: str | None = None,
    failure_context: str | None = None,
    repair_attempt: int = 0,
) -> str:
    """
    Request an updated policy using Cerebras via the OpenAI-compatible client.
    """
    frontier_context_rendered = frontier_context or format_frontier_context(frontier_context_details)
    try:
        WriterRequest(
            feedback=feedback or {},
            current_policy=current_policy,
            tests=tests,
            frontier_context=frontier_context_rendered,
            frontier_context_details=frontier_context_details,
            optimization_brief=optimization_brief,
            parent_trace=parent_trace,
            writer_session_id=writer_session_id,
            failure_context=failure_context,
            repair_attempt=repair_attempt,
        )
    except ValidationError as exc:
        raise RuntimeError(f"Invalid writer request: {exc}") from exc

    parent_span = _require_parent(parent_trace)
    client, model = _get_client()
    if client is None or not hasattr(client, "chat"):
        raise RuntimeError("Writer client is not available")
    model_budget = get_model_budget(model)
    failure_context_budget = compute_prompt_char_budget(model, repair_attempt=repair_attempt)
    failure_context = failure_context or ""
    if failure_context:
        failure_context = failure_context[:failure_context_budget]
    completion_tokens = model_budget.completion_reserve
    rendered_prompt = build_writer_prompt(
        current_policy=current_policy,
        failure_context=failure_context,
        tests=tests,
        frontier_context=frontier_context_rendered,
        frontier_context_details=frontier_context_details,
        optimization_brief=optimization_brief,
    )
    logger = bind_logger(
        _LOGGER,
        writer_session_id=writer_session_id,
        repair_attempt=repair_attempt,
        model=model,
    )
    with parent_span.start_as_current_observation(
        name="policy_writer",
        as_type="span",
        metadata=serialize_observation_metadata(
            {
                "model": model,
                "writer_session_id": writer_session_id,
                "session_id": writer_session_id,
            }
            if writer_session_id
            else {"model": model}
        ),
    ) as writer_obs:
        response = None
        logger.info(
            "policy_writer_started",
            completion_tokens=completion_tokens,
            failure_context_budget=failure_context_budget,
            prompt_chars=len(rendered_prompt.system) + len(rendered_prompt.user),
        )
        try:
            writer_obs.score(
                name="writer.policy_generated",
                value=0,
                data_type=ScoreDataType.BOOLEAN,
            )
            writer_obs.score(
                name="writer.policy_compiled",
                value=0,
                data_type=ScoreDataType.BOOLEAN,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "langfuse_score_emission_failed",
                operation="policy_writer.score",
                observation=_span_label(writer_obs),
                score="writer.policy_generated|writer.policy_compiled",
                error=str(exc),
            )
        for attempt in range(3):
            start_time = time.perf_counter()
            attempt_logger = bind_logger(logger, writer_attempt=attempt)
            attempt_logger.info("writer_attempt_started")
            with writer_obs.start_as_current_observation(
                name=f"writer_attempt_{attempt}",
                as_type="generation",
                model=model,
                metadata=serialize_observation_metadata(
                    {
                        "attempt": attempt,
                        "model": model,
                        "writer_session_id": writer_session_id,
                        "session_id": writer_session_id,
                    }
                    if writer_session_id
                    else {"attempt": attempt, "model": model}
                ),
            ) as attempt_obs:
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": rendered_prompt.system,
                        },
                        {
                            "role": "user",
                            "content": rendered_prompt.user,
                        },
                    ]
                    _ensure_non_empty_messages(messages, caller="writer")
                    response = client.chat.completions.create(
                        model=model,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "policy_code",
                                "schema": {
                                    "type": "object",
                                    "properties": {"code": {"type": "string"}},
                                    "required": ["code"],
                                    "additionalProperties": False,
                                },
                                "strict": True,
                            },
                        },
                        max_completion_tokens=completion_tokens,
                        messages=messages,
                    )
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    usage_env = usage_from_response(response)
                    usage_details: dict[str, object] | None = (
                        {k: v for k, v in usage_env.model_dump_flat().items()} if usage_env else None
                    )
                    if usage_details is None:
                        raise RuntimeError("Cerebras response missing usage; provide usage_details explicitly or enable provider usage output.")
                    from harness.telemetry.tracing.cerebras_pricing import cost_details_for_usage

                    cost_details = cost_details_for_usage(model, usage_details) if usage_details else None
                    try:
                        attempt_obs.update(
                            metadata=serialize_observation_metadata(
                                {"model": model, "attempt": attempt, "latency_ms": latency_ms}
                            ),
                            usage_details=_langfuse_usage_details(usage_details),
                            cost_details=dict(cost_details) if cost_details is not None else None,
                        )
                    except Exception as exc:  # noqa: BLE001
                        _LOGGER.warning(
                            "langfuse_metadata_update_failed",
                            operation="writer_attempt.update",
                            observation=_span_label(attempt_obs),
                            keys=["attempt", "latency_ms", "model"],
                            error=str(exc),
                        )
                    attempt_logger.info(
                        "writer_attempt_completed",
                        latency_ms=latency_ms,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    attempt_logger.warning(
                        "writer_attempt_failed",
                        error=str(e),
                    )
                    try:
                        attempt_obs.update(
                            metadata=serialize_observation_metadata({"error": str(e)})
                        )
                    except Exception as update_exc:  # noqa: BLE001
                        _LOGGER.warning(
                            "langfuse_metadata_update_failed",
                            operation="writer_attempt.update",
                            observation=_span_label(attempt_obs),
                            keys=["error"],
                            error=str(update_exc),
                        )
                    time.sleep(2 * (attempt + 1))
        else:
            logger.warning("writer_attempt_exhausted", retries=3)
            try:
                writer_obs.update(
                    metadata=serialize_observation_metadata({"error": "writer_failed"})
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "langfuse_metadata_update_failed",
                    operation="policy_writer.update",
                    observation=_span_label(writer_obs),
                    keys=["error"],
                    error=str(exc),
                )
            raise RuntimeError("Writer failed after retries")

        try:
            raw_code = _extract_policy_code(response)
            code = _ensure_valid_policy_code(raw_code)
            logger.info(
                "writer_compile_validation_passed",
                code_chars=len(code),
            )
            try:
                writer_obs.score(
                    name="writer.policy_compiled",
                    value=1,
                    data_type=ScoreDataType.BOOLEAN,
                    comment=json.dumps({"model": model}),
                )
            except Exception as score_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "langfuse_score_emission_failed",
                    operation="policy_writer.score",
                    observation=_span_label(writer_obs),
                    score="writer.policy_compiled",
                    error=str(score_exc),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "writer_compile_validation_failed",
                error=str(exc),
            )
            try:
                writer_obs.update(
                    metadata=serialize_observation_metadata({"error": str(exc), "model": model})
                )
            except Exception as update_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "langfuse_metadata_update_failed",
                    operation="policy_writer.update",
                    observation=_span_label(writer_obs),
                    keys=["error", "model"],
                    error=str(update_exc),
                )
            raise
        try:
            writer_obs.score(
                name="writer.policy_generated",
                value=1,
                data_type=ScoreDataType.BOOLEAN,
                comment=json.dumps({"model": model}),
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "langfuse_score_emission_failed",
                operation="policy_writer.score",
                observation=_span_label(writer_obs),
                score="writer.policy_generated",
                error=str(exc),
            )
        try:
            writer_obs.update(
                metadata=serialize_observation_metadata(
                    {
                        "model": model,
                        "length": len(code),
                        "feedback_keys": list((feedback or {}).keys()),
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "langfuse_metadata_update_failed",
                operation="policy_writer.update",
                observation=_span_label(writer_obs),
                keys=["feedback_keys", "length", "model"],
                error=str(exc),
            )
        logger.info(
            "policy_writer_completed",
            code_chars=len(code),
            feedback_keys=sorted((feedback or {}).keys()),
        )
        return code
