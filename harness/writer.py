from __future__ import annotations

"""
Leaf operation: policy writer/generator.
No CLI or orchestration logic; only generation and validation.
"""

import json
import time
from typing import Any, Mapping

from .observability.langfuse import get_tracing_openai_client, start_observation
from .observability.score_emitter import emit_score_for_handle
from .utils.env import get_cerebras_api_key, get_writer_model
from .utils.model_budgets import compute_prompt_char_budget, get_model_budget
from .prompts import WriterPromptContext
from .utils.template_loader import render_prompt_template
from .runner import _usage_from_response  # reuse usage mapping for OpenAI-style responses  # pyright: ignore[reportPrivateUsage]

_client: Any | None = None
_WRITER_TEMPLATE = "policy_writer.jinja"


def _get_client():
    global _client  # noqa: PLW0603
    if _client is None:
        api_key = get_cerebras_api_key()
        if not api_key:
            raise RuntimeError("Missing CEREBRAS_API_KEY in environment")
        _client = get_tracing_openai_client(base_url="https://api.cerebras.ai/v1", api_key=api_key)
    return _client, get_writer_model()


def _extract_policy_code(response: Any) -> str:
    """
    Extract structured code from a model response.

    Supports:
    - response.choices[0].message.content as JSON string (primary)
    - response.choices[0].message.content as mapping
    - response.choices[0].message.parsed (mapping with "code") as compatibility fallback
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

    parsed = getattr(msg, "parsed", None)
    if isinstance(parsed, Mapping):
        code_val = _mapping_code(parsed)
        if code_val is not None:
            return code_val

    raise ValueError("Writer response missing content")


def _ensure_valid_policy_code(code: str) -> str:
    cleaned = code if isinstance(code, str) else ""
    if not cleaned or not cleaned.strip():
        raise ValueError("Writer returned empty code")
    if "```" in cleaned:
        raise ValueError("Writer returned fenced code")
    if "def score" not in cleaned:
        raise ValueError("Writer missing score function")
    compile(cleaned, "<generated-policy>", "exec")
    return cleaned


def _render_writer_prompt(
    *,
    current_policy: str,
    failure_context: str,
    feedback_str: str,
    feedback: Mapping[str, object],
    comparison_summary: str | None,
    guidance_hint: str,
    diff_str: str,
    diff_hint: str,
    tests: str,
    scoring_context: str,
) -> str:
    feedback_text = feedback_str if feedback_str else str(feedback)
    context = WriterPromptContext(
        current_policy=current_policy,
        failure_context=failure_context,
        feedback_text=feedback_text,
        comparison_summary=comparison_summary or "N/A",
        guidance_hint=guidance_hint,
        diff_str=diff_str,
        diff_hint=diff_hint,
        tests=tests,
        scoring_context=scoring_context,
    )
    rendered = render_prompt_template(_WRITER_TEMPLATE, context.model_dump())
    if "===USER===" not in rendered:
        rendered = f"===USER===\n{rendered}"
    return rendered


def generate_policy(
    feedback: dict[str, Any],
    current_policy: str,
    tests: str,
    scoring_context: str,
    feedback_str: str,
    guidance_hint: str,
    diff_str: str,
    diff_hint: str,
    comparison_summary: str | None = None,
    parent_trace: object | None = None,
    failure_context: str | None = None,
    repair_attempt: int = 0,
) -> str:
    """
    Request an updated policy using Cerebras via the OpenAI-compatible client.
    """
    client, model = _get_client()
    if client is None or not hasattr(client, "chat"):
        raise RuntimeError("Writer client is not available")
    model_budget = get_model_budget(model)
    failure_context_budget = compute_prompt_char_budget(model, repair_attempt=repair_attempt)
    failure_context = failure_context or ""
    if failure_context:
        failure_context = failure_context[:failure_context_budget]
    completion_tokens = model_budget.completion_reserve
    prompt_content = _render_writer_prompt(
        current_policy=current_policy,
        failure_context=failure_context,
        feedback_str=str(feedback_str),
        feedback=feedback,
        comparison_summary=comparison_summary,
        guidance_hint=guidance_hint,
        diff_str=diff_str,
        diff_hint=diff_hint,
        tests=tests,
        scoring_context=scoring_context,
    )
    with start_observation(
        name="policy_writer",
        parent=parent_trace,
        metadata={"model": model},
    ) as writer_obs:
        response = None
        emit_score_for_handle(writer_obs, name="writer.policy_generated", value=False, data_type="BOOLEAN")
        emit_score_for_handle(writer_obs, name="writer.policy_compiled", value=False, data_type="BOOLEAN")
        for attempt in range(3):
            start_time = time.perf_counter()
            with start_observation(
                name=f"writer_attempt_{attempt}",
                parent=writer_obs,
                as_type="generation",
                model=model,
                metadata={"attempt": attempt, "model": model},
            ) as attempt_obs:
                try:
                    system_content, user_content = prompt_content.split("===USER===", 1)
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
                        messages=[
                            {
                                "role": "system",
                                "content": system_content.strip(),
                            },
                            {
                                "role": "user",
                                "content": user_content.strip(),
                            },
                        ],
                    )
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    usage_details = _usage_from_response(response)
                    if usage_details is None:
                        raise RuntimeError("Cerebras response missing usage; provide usage_details explicitly or enable provider usage output.")
                    from .observability.cerebras_pricing import cost_details_for_usage

                    cost_details = cost_details_for_usage(model, usage_details) if usage_details else None
                    attempt_obs.update(
                        metadata={"model": model, "attempt": attempt, "latency_ms": latency_ms},
                        usage_details=usage_details,
                        cost_details=cost_details,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    attempt_obs.end(error=e)
                    time.sleep(2 * (attempt + 1))
        else:
            writer_obs.end(error="writer_failed")
            raise RuntimeError("Writer failed after retries")

        try:
            raw_code = _extract_policy_code(response)
            code = _ensure_valid_policy_code(raw_code)
            emit_score_for_handle(
                writer_obs,
                name="writer.policy_compiled",
                value=True,
                data_type="BOOLEAN",
                comment=json.dumps({"model": model}),
            )
        except Exception as exc:  # noqa: BLE001
            writer_obs.end(error=exc, metadata={"model": model})
            raise
        emit_score_for_handle(
            writer_obs,
            name="writer.policy_generated",
            value=True,
            data_type="BOOLEAN",
            comment=json.dumps({"model": model}),
        )
        writer_obs.update(metadata={"model": model, "length": len(code), "feedback_keys": list(feedback.keys())})
        return code
