"""
Localizer agent: LLM/tool-execution layer for iterative-context and jcodemunch
bug-localization flows.  Owns message construction, model calls, tool dispatch,
context-budget management, finalization, and retry logic.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Callable, Literal, cast

from langfuse import LangfuseGeneration, LangfuseSpan
from langfuse.api.commons.types import ScoreDataType

from harness.log import bind_logger, get_logger, short_task_label
from harness.telemetry.tracing import (
    UsageDetails,
    get_tracing_openai_client,
    serialize_observation_metadata,
)
from harness.telemetry.observability_models import TaskRunSummary, TokenUsageBreakdown
from harness.prompts import build_ic_system_prompt, build_jc_system_prompt
from harness.backends.ic import IterativeContextBackend
from harness.backends.jc import JCodeMunchBackend
from harness.backends.mcp import serialize_tool_result_for_model
from harness.utils.env import get_cerebras_api_key, get_runner_model, getenv
from harness.utils.openai_schema import ChatCompletionToolParam, validate_tools

from harness.localization.models import LCATask, LocalizationPrediction, LocalizationRunResult
from harness.localization.errors import LocalizationEvaluationError, LocalizationFailureCategory

from harness.agents.common import normalize_token_usage, serialize_lca_task_for_prompt, usage_from_response

_MAX_TOTAL_MESSAGE_CHARS = 24000
_MAX_TOOL_CONTENT_CHARS = 4000
_MAX_SYSTEM_CHARS = 4000
_MAX_USER_CHARS = 4000
_HISTORY_LIMIT = 3
_AGGRESSIVE_TOTAL_CHARS = 12000
_AGGRESSIVE_TOOL_CONTENT_CHARS = 1200
_AGGRESSIVE_SYSTEM_CHARS = 1500
_AGGRESSIVE_USER_CHARS = 1500

_LOGGER = get_logger(__name__)


def _require_parent(
    parent: LangfuseSpan | LangfuseGeneration | None,
    owner: str = "runner agent",
) -> LangfuseSpan | LangfuseGeneration:
    if parent is None:
        raise RuntimeError(f"Parent span required for {owner} tracing")
    return parent


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


def _prediction_from_output(
    predicted_files: object,
    reasoning: object = None,
) -> LocalizationPrediction:
    normalized: list[str] = []
    seen: set[str] = set()
    candidates: list[object]
    if isinstance(predicted_files, str):
        candidates = list(predicted_files.replace("\n", ",").split(","))
    elif isinstance(predicted_files, list):
        candidates = predicted_files
    else:
        candidates = []
    for item in candidates:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return LocalizationPrediction(
        predicted_files=normalized,
        reasoning=reasoning if isinstance(reasoning, str) else None,
    )


def _sanitize_agent_task(agent_task: Mapping[str, object]) -> dict[str, object]:
    """Drop gold/irrelevant fields before showing the payload to the model."""
    cleaned: dict[str, object] = {}
    for key, val in agent_task.items():
        cleaned[key] = val
    return cleaned


def _format_user_prompt(agent_task: dict[str, object]) -> str:
    payload_json = json.dumps(_sanitize_agent_task(agent_task), indent=2)
    return (
        "Localize the bug by using the available tools and propose repo-relative files to change.\n"
        "Prefer concise file lists (JSON or bullets are fine) plus brief reasoning.\n"
        "Task payload:\n"
        f"{payload_json}"
    )


def _truncate_text(text: str, max_length: int, keep_tail: int = 400) -> str:
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    tail = min(keep_tail, max_length // 4)
    head = max_length - tail - 24  # padding for marker
    if head < 0:
        head = max_length
        tail = 0
    truncated = text[:head]
    suffix = text[-tail:] if tail > 0 else ""
    removed = len(text) - len(truncated) - len(suffix)
    marker = f"...[truncated {removed} chars]..."
    return f"{truncated}{marker}{suffix}"[:max_length]


def _compact_tool_content(result: Any, limit: int = _MAX_TOOL_CONTENT_CHARS) -> str:
    serialized = serialize_tool_result_for_model(result)
    if not isinstance(serialized, str):
        serialized = str(serialized)
    return _truncate_text(serialized, limit)


def _message_content_length(msg: Mapping[str, object]) -> int:
    content = msg.get("content")
    try:
        if isinstance(content, str):
            return len(content)
        return len(json.dumps(content))
    except Exception:
        return len(str(content))


def _enforce_message_budget(
    messages: list[dict[str, object]],
    max_total_chars: int = _MAX_TOTAL_MESSAGE_CHARS,
    max_tool_content: int = _MAX_TOOL_CONTENT_CHARS,
    max_system_chars: int = _MAX_SYSTEM_CHARS,
    max_user_chars: int = _MAX_USER_CHARS,
    keep_last_n_tools: int = _HISTORY_LIMIT,
) -> list[dict[str, object]]:
    trimmed: list[dict[str, object]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        new_msg = dict(msg)
        if isinstance(content, str):
            if role == "system":
                new_msg["content"] = _truncate_text(content, max_system_chars)
            elif role == "user":
                new_msg["content"] = _truncate_text(content, max_user_chars)
            elif role == "tool":
                new_msg["content"] = _truncate_text(content, max_tool_content)
        trimmed.append(new_msg)

    if len(trimmed) <= 2:
        return trimmed

    system_msg = trimmed[0]
    user_msg = trimmed[1]
    tool_msgs = trimmed[2:]
    total_len = _message_content_length(system_msg) + _message_content_length(user_msg)

    kept_tools: list[dict[str, object]] = []
    for msg in reversed(tool_msgs):
        if len(kept_tools) >= keep_last_n_tools:
            break
        kept_tools.append(msg)
        total_len += _message_content_length(msg)
        if total_len >= max_total_chars:
            break

    kept_tools.reverse()
    compacted = [system_msg, user_msg] + kept_tools

    total_len = sum(_message_content_length(m) for m in compacted)
    if total_len > max_total_chars:
        compacted[0]["content"] = _truncate_text(str(compacted[0].get("content", "")), max_system_chars // 2)
        compacted[1]["content"] = _truncate_text(str(compacted[1].get("content", "")), max_user_chars // 2)
        for idx, msg in enumerate(compacted[2:], start=2):
            msg_content = str(msg.get("content", ""))
            compacted[idx]["content"] = _truncate_text(msg_content, max_tool_content // 2)
    return compacted


def _aggressive_compact_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    return _enforce_message_budget(
        messages,
        max_total_chars=_AGGRESSIVE_TOTAL_CHARS,
        max_tool_content=_AGGRESSIVE_TOOL_CONTENT_CHARS,
        max_system_chars=_AGGRESSIVE_SYSTEM_CHARS,
        max_user_chars=_AGGRESSIVE_USER_CHARS,
        keep_last_n_tools=1,
    )


def _is_context_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "context" in msg and "length" in msg:
        return True
    if "reduce the length" in msg or "too long" in msg:
        return True
    status = getattr(exc, "status_code", None)
    return status == 400 and ("length" in msg or "context" in msg)


def _truncate_for_metadata(payload: object, limit: int = 2000) -> object:
    try:
        serialized = json.dumps(payload)
        if len(serialized) <= limit:
            return payload
        return f"{serialized[: limit - 4]}..."
    except Exception:
        text = str(payload)
        return f"{text[: limit - 4]}..." if len(text) > limit else payload


def resolve_runner_model(model_override: str | None = None) -> str:
    """
    Resolve the runner model, preferring explicit overrides and falling back to env default.
    """
    env_model = getenv("RUNNER_MODEL")
    resolved = model_override or env_model or get_runner_model()
    if not resolved:
        resolved = "gpt-oss-120b"
    _LOGGER.info(
        "runner_model_resolved",
        model=resolved,
        override=bool(model_override),
        env_model=env_model,
    )
    return resolved


def _make_client(model_override: str | None = None) -> tuple[Any, str]:
    api_key = get_cerebras_api_key()
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is required for runner agent client")
    model_name = resolve_runner_model(model_override)
    return get_tracing_openai_client(base_url="https://api.cerebras.ai/v1", api_key=api_key), model_name


def _build_model_messages(agent_task: dict[str, object], tool_specs: list[ChatCompletionToolParam], system_prompt: str) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _format_user_prompt(agent_task),
        },
    ]
    validate_tools(tool_specs)
    return messages


def _ensure_non_empty_messages(messages: list[dict[str, object]] | None, caller: str) -> None:
    if not messages:
        raise RuntimeError(f"{caller}: empty messages payload")


def _collect_tool_results(
    observations: list[dict[str, object]],
    dispatch_tool_call: Callable[[str, dict[str, Any]], object],
    budget_limit: int,
    parent_trace: LangfuseSpan | LangfuseGeneration | None = None,
) -> list[dict[str, object]]:
    parent_span = _require_parent(parent_trace, owner="toolcall")
    total_chars = sum(_message_content_length(obs) for obs in observations)
    if total_chars > budget_limit:
        observations = _aggressive_compact_messages(observations)
    tool_calls = [obs["tool_call"] for obs in observations if isinstance(obs, Mapping) and "tool_call" in obs]
    new_results = []
    for call in tool_calls:
        call_map: Mapping[str, object] | None = call if isinstance(call, Mapping) else None
        if call_map is None:
            func = getattr(call, "function", None)
            name = getattr(func, "name", None) or getattr(call, "name", None)
            arguments = getattr(func, "arguments", None) or getattr(call, "arguments", None)
            call_map = {"name": name, "id": getattr(call, "id", None), "arguments": arguments} if name else None
        if not isinstance(call_map, Mapping):
            continue
        name = cast(str | None, call_map.get("name"))
        arguments = call_map.get("arguments")
        if name:
            arg_map = arguments if isinstance(arguments, dict) else {}
            if isinstance(arguments, str):
                try:
                    loaded_args = json.loads(arguments)
                    if isinstance(loaded_args, dict):
                        arg_map = loaded_args
                except Exception:
                    pass
            with parent_span.start_as_current_observation(
                name=f"toolcall.{name}",
                as_type="span",
                metadata=serialize_observation_metadata({"tool": name}),
            ) as tool_obs:
                try:
                    result = dispatch_tool_call(name, arg_map)
                    failed = False
                except Exception as exc:  # noqa: BLE001
                    result = {"error": str(exc)}
                    failed = True
                new_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_map.get("id"),
                        "content": _compact_tool_content(result),
                        "result": result,
                        "tool_name": name,
                        "tool_status": "failure" if failed else "success",
                    }
                )
                try:
                    tool_obs.update(
                        output=_truncate_for_metadata(result),
                        metadata=serialize_observation_metadata(
                            {
                                "tool": name,
                                "phase": "tool_execution",
                                "status": "failure" if failed else "success",
                            }
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "langfuse_metadata_update_failed",
                        operation="toolcall.update",
                        observation=_span_label(tool_obs),
                        keys=["phase", "status", "tool"],
                        error=str(exc),
                    )
    return new_results


def _make_retry_messages(
    agent_task: dict[str, object],
    observations: list[dict[str, object]],
    system_prompt: str,
    tool_specs: list[ChatCompletionToolParam],
    aggressive: bool,
) -> list[dict[str, object]]:
    base_messages = _build_model_messages(agent_task, tool_specs, system_prompt)
    if aggressive:
        return _aggressive_compact_messages(base_messages + observations)
    return _enforce_message_budget(base_messages + observations)


def _apply_tool_results(
    agent_task: dict[str, object],
    observations: list[dict[str, object]],
    tool_results: list[dict[str, object]],
    system_prompt: str,
    tool_specs: list[ChatCompletionToolParam],
) -> list[dict[str, object]]:
    ordered_obs = list(observations)
    for result in tool_results:
        if result not in ordered_obs:
            ordered_obs.append(result)
    return _enforce_message_budget(
        _build_model_messages(agent_task, tool_specs, system_prompt) + ordered_obs,
        max_total_chars=_AGGRESSIVE_TOTAL_CHARS,
        max_tool_content=_AGGRESSIVE_TOOL_CONTENT_CHARS,
        max_system_chars=_AGGRESSIVE_SYSTEM_CHARS,
        max_user_chars=_AGGRESSIVE_USER_CHARS,
        keep_last_n_tools=max(1, len(ordered_obs)),
    )


def _collect_usage_details(response: Any) -> dict[str, object] | None:
    usage_envelope = usage_from_response(response)
    usage_details: dict[str, object] | None = (
        {k: v for k, v in usage_envelope.model_dump_flat().items()} if usage_envelope else None
    )
    if not usage_details:
        return None
    model = getattr(response, "model", None)
    if isinstance(model, str):
        usage_details["model"] = model
    try:
        usage_model = UsageDetails.model_validate(usage_details)
    except Exception:
        return usage_details
    return usage_model.to_payload()


_USAGE_TOTAL_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input",
    "input_cached_tokens",
    "prompt_tokens_details.cached_tokens",
    "output",
    "total",
)


def _accumulate_usage_details(
    aggregate: dict[str, object],
    usage_details: Mapping[str, object],
) -> None:
    for key in _USAGE_TOTAL_KEYS:
        value = usage_details.get(key)
        if isinstance(value, (int, float)):
            existing = aggregate.get(key, 0.0)
            existing_total = float(existing) if isinstance(existing, (int, float)) else 0.0
            aggregate[key] = existing_total + float(value)
    model = usage_details.get("model")
    if isinstance(model, str) and model:
        aggregate["model"] = model


def run_agent(
    agent_task: dict[str, object],
    steps: int,
    tool_specs: list[ChatCompletionToolParam],
    dispatch_tool_call: Callable[[str, dict[str, Any]], object],
    system_prompt: str,
    parent_trace: LangfuseSpan | LangfuseGeneration | None = None,
    backend_name: str | None = None,
):
    parent_span = _require_parent(parent_trace, owner=backend_name or "runner agent")
    client, model = _make_client()
    try:
        parent_span.update(metadata=serialize_observation_metadata({"runner_model": model}))
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "langfuse_metadata_update_failed",
            operation="runner_parent.update",
            observation=_span_label(parent_span),
            keys=["runner_model"],
            error=str(exc),
        )
    messages = _build_model_messages(agent_task, tool_specs, system_prompt)
    tool_choice = "required" if tool_specs else None
    tools_payload: list[ChatCompletionToolParam] | None = tool_specs or None
    observations: list[dict[str, object]] = []
    aggregate_usage_details: dict[str, object] = {}
    step_count = 0
    generation_count = 0
    run_logger = bind_logger(
        _LOGGER,
        backend=backend_name or "localizer",
    )

    for step in range(steps):
        step_count += 1
        step_logger = bind_logger(
            run_logger,
            step=step,
            step_ordinal=step + 1,
            steps=steps,
            model=model,
        )
        step_logger.info("agent_step_started")
        with parent_span.start_as_current_observation(
            name="agent_step",
            as_type="span",
            metadata=serialize_observation_metadata(
                {"step": step, "model": model, "backend": backend_name or "unknown"}
            ),
        ) as obs:
            tool_results = _collect_tool_results(
                observations,
                dispatch_tool_call=dispatch_tool_call,
                budget_limit=_MAX_TOTAL_MESSAGE_CHARS,
                parent_trace=obs,
            )
            if tool_results:
                observations.extend(tool_results)
            messages = _apply_tool_results(agent_task, observations, tool_results, system_prompt, tool_specs)
            try:
                if not messages:
                    messages = _build_model_messages(agent_task, tool_specs, system_prompt)
                _ensure_non_empty_messages(messages, caller=backend_name or "runner agent")
                generation_count += 1
                completion = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.2,
                    tools=tools_payload,
                    tool_choice=tool_choice,
                    extra_headers={"x-cerebras-api-key": get_cerebras_api_key()},
                )
            except Exception as exc:  # noqa: BLE001
                if not _is_context_error(exc):
                    raise
                retry_messages = _make_retry_messages(
                    agent_task,
                    observations,
                    system_prompt,
                    tool_specs,
                    aggressive=True,
                )
                if not retry_messages:
                    retry_messages = _build_model_messages(agent_task, tool_specs, system_prompt)
                _ensure_non_empty_messages(retry_messages, caller=backend_name or "runner agent retry")
                generation_count += 1
                completion = client.chat.completions.create(
                    messages=retry_messages,
                    model=model,
                    temperature=0.2,
                    tools=tools_payload,
                    tool_choice=tool_choice,
                    extra_headers={"x-cerebras-api-key": get_cerebras_api_key()},
                )
            choice = completion.choices[0]
            usage_details = _collect_usage_details(completion)
            if usage_details:
                _accumulate_usage_details(aggregate_usage_details, usage_details)
                obs.update(usage_details=_langfuse_usage_details(usage_details))
            else:
                raise RuntimeError("Agent response missing usage details")
            message_obj = getattr(choice, "message", None)
            if isinstance(message_obj, Mapping):
                response_message = dict(message_obj)
            else:
                tool_calls_attr = getattr(message_obj, "tool_calls", None)
                content_attr = getattr(message_obj, "content", None)
                response_message = {}
                if tool_calls_attr is not None:
                    response_message["tool_calls"] = tool_calls_attr
                if content_attr is not None:
                    response_message["content"] = content_attr
            tool_calls_raw = response_message.get("tool_calls")
            tool_calls: list[object] = []
            if isinstance(tool_calls_raw, list):
                tool_calls = list(tool_calls_raw)
            elif tool_calls_raw is not None:
                tool_calls = [tool_calls_raw]
            if tool_calls:
                tool_names: list[str] = []
                for call in tool_calls:
                    function_obj = getattr(call, "function", None)
                    name_obj = getattr(function_obj, "name", None)
                    if isinstance(name_obj, str):
                        tool_names.append(name_obj)
                step_logger.info(
                    "agent_step_tool_calls",
                    tool_calls=tool_names or ["unknown"],
                )
                assistant_observation = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                    "content": response_message.get("content") or "",
                    "usage_details": usage_details,
                }
                observations.append(assistant_observation)
                tool_results = _collect_tool_results(
                    observations=[{"tool_call": call} for call in tool_calls],
                    dispatch_tool_call=dispatch_tool_call,
                    budget_limit=_MAX_TOTAL_MESSAGE_CHARS,
                    parent_trace=obs,
                )
                if tool_results:
                    observations.extend(tool_results)
                messages = _apply_tool_results(agent_task, observations, [], system_prompt, tool_specs)
                continue
            else:
                raw_content = response_message.get("content")
                content_text = raw_content if isinstance(raw_content, str) else "{}"
                try:
                    response_json = json.loads(content_text)
                except json.JSONDecodeError as exc:  # noqa: BLE001
                    response_json = {"error": f"invalid json: {exc}"}
                obs.update(output=_truncate_for_metadata(response_json))
                observations.append(
                    {
                        "role": "assistant",
                        "content": response_json,
                        "usage_details": usage_details,
                    }
                )
                predicted = response_json.get("predicted_files")
                predicted_count = len(predicted) if isinstance(predicted, list) else 0
                step_logger.info(
                    "agent_step_completed",
                    predicted_files_count=predicted_count,
                )
                break
    result: dict[str, object] = {"observations": observations}
    if aggregate_usage_details:
        result["usage_details"] = aggregate_usage_details
    result["step_count"] = step_count
    result["generation_count"] = generation_count
    return result


def run_agent_with_retry(
    agent_task: dict[str, object],
    steps: int,
    tool_specs: list[ChatCompletionToolParam],
    dispatch_tool_call: Callable[[str, dict[str, Any]], object],
    system_prompt: str,
    parent_trace: LangfuseSpan | LangfuseGeneration | None = None,
    backend_name: str | None = None,
):
    parent_span = _require_parent(parent_trace, owner=backend_name or "runner agent")
    try:
        return run_agent(
            agent_task,
            steps,
            tool_specs,
            dispatch_tool_call,
            system_prompt,
            parent_trace=parent_span,
            backend_name=backend_name,
        )
    except Exception as exc:  # noqa: BLE001
        if not _is_context_error(exc):
            raise
        retry_messages = _make_retry_messages(agent_task, [], system_prompt, tool_specs, aggressive=True)
        if not retry_messages:
            retry_messages = _build_model_messages(agent_task, tool_specs, system_prompt)
        client, model = _make_client()
        _ensure_non_empty_messages(retry_messages, caller=backend_name or "runner agent retry")
        completion = client.chat.completions.create(
            messages=retry_messages,
            model=model,
            temperature=0.2,
            tools=tool_specs or None,
            tool_choice="required" if tool_specs else None,
            extra_headers={"x-cerebras-api-key": get_cerebras_api_key()},
        )
        choice = completion.choices[0]
        response_message = dict(choice.message) if hasattr(choice, "message") else {}
        try:
            response_json = json.loads(response_message.get("content") or "{}")
        except json.JSONDecodeError as parse_exc:  # noqa: BLE001
            response_json = {"error": f"invalid json: {parse_exc}"}
        # Ensure a minimal tool-like message exists to satisfy downstream expectations.
        tool_msg = {"role": "tool", "content": response_json}
        return {"observations": [tool_msg, {"role": "assistant", "content": response_json}]}


def _normalize_agent_result(raw: Any) -> dict[str, object]:
    observations_raw = raw.get("observations") if isinstance(raw, Mapping) else []
    observations = observations_raw if isinstance(observations_raw, list) else []

    def _latest_graph_node_count() -> int:
        # Prefer explicit node_count on raw if provided.
        if isinstance(raw, Mapping):
            raw_node_count = raw.get("node_count")
            if isinstance(raw_node_count, (int, float)):
                return int(raw_node_count)
        for latest in reversed(observations):
            if not isinstance(latest, Mapping):
                continue
            result = latest.get("result")
            if not isinstance(result, Mapping):
                continue
            full_graph = result.get("full_graph")
            if isinstance(full_graph, Mapping):
                nodes = full_graph.get("nodes")
                if isinstance(nodes, list):
                    return len(nodes)
            graph = result.get("graph")
            if isinstance(graph, Mapping):
                nodes = graph.get("nodes")
                if isinstance(nodes, list):
                    return len(nodes)
        return 0

    normalized: dict[str, object] = {
        "observations": observations,
        "node_count": _latest_graph_node_count(),
        "raw": raw,
    }
    # Preserve any graph/full_graph for downstream consumers.
    if isinstance(raw, Mapping):
        if "full_graph" in raw:
            normalized["full_graph"] = raw.get("full_graph")
        if "graph" in raw:
            normalized["graph"] = raw.get("graph")
    if isinstance(raw, dict) and "error" in raw:
        normalized["error"] = raw["error"]
    return normalized


def _coerce_observations(value: object) -> list[dict[str, object]]:
    """
    Ensure observations are stored as a list of dicts for downstream consumers and type safety.
    Non-mapping entries are discarded because they cannot be serialized consistently.
    """
    observations: list[dict[str, object]] = []
    if isinstance(value, list):
        for obs in value:
            if isinstance(obs, Mapping):
                observations.append(dict(obs))
    return observations


def _coerce_node_count(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _coerce_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0


def _observation_token_breakdown(observations: list[dict[str, object]]) -> TokenUsageBreakdown:
    usage_entries: list[TokenUsageBreakdown] = []
    for obs in observations:
        usage_details = obs.get("usage_details")
        if isinstance(usage_details, Mapping):
            usage_entries.append(normalize_token_usage(usage_details))
    return TokenUsageBreakdown.sum(usage_entries)


def _raw_token_breakdown(raw: object) -> TokenUsageBreakdown:
    if not isinstance(raw, Mapping):
        return TokenUsageBreakdown()
    usage_details = raw.get("usage_details")
    return normalize_token_usage(usage_details if isinstance(usage_details, Mapping) else None)


def _raw_generation_count(raw: object) -> int:
    if not isinstance(raw, Mapping):
        return 0
    return _coerce_count(raw.get("generation_count"))


def _tool_stats(observations: list[dict[str, object]]) -> tuple[int, int, int]:
    total = 0
    failed = 0
    for obs in observations:
        if obs.get("role") != "tool":
            continue
        total += 1
        result = obs.get("result")
        status = obs.get("tool_status")
        if status == "failure" or (
            isinstance(result, Mapping) and isinstance(result.get("error"), (str, bytes))
        ):
            failed += 1
    successful = total - failed
    return total, successful, failed


def _build_task_run_summary(
    *,
    task: LCATask,
    backend: Literal["ic", "jc"],
    status: Literal["success", "failure"],
    started_at: float,
    raw: object,
    observations: list[dict[str, object]],
    predicted_files_count: int = 0,
    stage: str | None = None,
    message: str | None = None,
) -> TaskRunSummary:
    agent_raw = raw.get("raw") if isinstance(raw, Mapping) else None
    raw_map = raw if isinstance(raw, Mapping) else {}
    tool_total, tool_success, tool_failed = _tool_stats(observations)
    tokens = _observation_token_breakdown(observations)
    if tokens.total_tokens <= 0 and isinstance(agent_raw, Mapping):
        tokens += _raw_token_breakdown(agent_raw)
    if isinstance(raw, Mapping) and (
        "final" in raw_map or (tokens.total_tokens <= 0 and "usage_details" in raw_map)
    ):
        tokens += _raw_token_breakdown(raw)
    return TaskRunSummary(
        backend=backend,
        identity=task.identity.task_id(),
        repo=task.repo or "",
        status=status,
        stage=stage,
        message=message,
        step_count=_coerce_count(raw_map.get("step_count"))
        + _coerce_count(agent_raw.get("step_count") if isinstance(agent_raw, Mapping) else None),
        generation_count=_coerce_count(raw_map.get("generation_count")) + _raw_generation_count(agent_raw),
        tool_calls_count=tool_total,
        successful_tool_calls_count=tool_success,
        failed_tool_calls_count=tool_failed,
        predicted_files_count=predicted_files_count,
        duration_seconds=max(0.0, time.perf_counter() - started_at),
        tokens=tokens,
    )


def _preserve_agent_raw(final: LocalizationRunResult, raw: object) -> None:
    if not isinstance(raw, Mapping):
        return
    if isinstance(final.raw, Mapping):
        merged = dict(final.raw)
        merged.setdefault("raw", dict(raw))
        final.raw = merged
    else:
        final.raw = {"raw": dict(raw)}


def _attach_task_summary(
    result: LocalizationRunResult,
    *,
    task: LCATask,
    backend: Literal["ic", "jc"],
    started_at: float,
    stage: str | None = None,
    message: str | None = None,
) -> LocalizationRunResult:
    status: Literal["success", "failure"] = (
        "success" if result.prediction.predicted_files else "failure"
    )
    summary = _build_task_run_summary(
        task=task,
        backend=backend,
        status=status,
        started_at=started_at,
        raw=result.raw or {},
        observations=result.observations,
        predicted_files_count=len(result.prediction.normalized_predicted_files()),
        stage=stage if status == "failure" else None,
        message=message,
    )
    result.observability_summary = summary
    return result


def _strip_repo_root(text: str, repo_root: str | None) -> str:
    if not repo_root or not isinstance(text, str):
        return text
    root = repo_root.rstrip("/\\")
    if not root:
        return text
    if text.startswith(root + "/") or text.startswith(root + "\\"):
        return text[len(root) + 1 :]
    return text.replace(root + "/", "").replace(root + "\\", "")


def _error_only_payload(value: object) -> bool:
    return isinstance(value, Mapping) and set(value.keys()) == {"error"} and isinstance(value.get("error"), (str, bytes))


def _scrub_value_for_finalization(value: object, repo_root: str | None, seen_files: set[str]) -> object:
    if isinstance(value, str):
        return _strip_repo_root(value, repo_root)
    if isinstance(value, Mapping):
        cleaned: dict[str, object] = {}
        for key, val in value.items():
            if _error_only_payload(val):
                continue
            if isinstance(val, str) and (key in {"file", "path"} or key.endswith(("_file", "_path"))):
                rel = _strip_repo_root(val, repo_root)
                if rel in seen_files:
                    continue
                seen_files.add(rel)
                cleaned[key] = rel
                continue
            cleaned[key] = _scrub_value_for_finalization(val, repo_root, seen_files)
        return cleaned
    if isinstance(value, list):
        cleaned_list: list[object] = []
        for item in value:
            if _error_only_payload(item):
                continue
            cleaned_item = _scrub_value_for_finalization(item, repo_root, seen_files)
            if isinstance(cleaned_item, str) and cleaned_item in seen_files:
                continue
            cleaned_list.append(cleaned_item)
        return cleaned_list
    return value


def _clean_observations_for_finalization(
    observations: list[dict[str, object]], repo_root: str | None
) -> list[dict[str, object]]:
    cleaned: list[dict[str, object]] = []
    seen_files: set[str] = set()
    for obs in observations:
        if not isinstance(obs, Mapping):
            continue
        content = obs.get("content")
        result = obs.get("result")
        if _error_only_payload(content) and (result is None or _error_only_payload(result)):
            continue
        if _error_only_payload(result) and (content is None or _error_only_payload(content)):
            continue
        cleaned_obs = _scrub_value_for_finalization(obs, repo_root, seen_files)
        if isinstance(cleaned_obs, Mapping) and cleaned_obs:
            cleaned.append(dict(cleaned_obs))
    return cleaned


_FILE_KEY_NAMES = {
    "file",
    "files",
    "filename",
    "filenames",
    "filepath",
    "filepaths",
    "path",
    "paths",
    "predicted_file",
    "predicted_files",
    "relative_path",
    "relative_paths",
    "repo_path",
    "repo_paths",
}
_FILE_SUFFIXES = ("_file", "_files", "_path", "_paths", "_filename", "_filenames")


def _is_file_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower()
    return normalized in _FILE_KEY_NAMES or normalized.endswith(_FILE_SUFFIXES)


def _normalize_candidate_file(value: str, repo_root: str | None) -> str | None:
    candidate = value.strip().strip("\"'")
    if not candidate:
        return None
    if "\n" in candidate or "\r" in candidate or "://" in candidate:
        return None
    if candidate.startswith(("git@", "#")):
        return None
    candidate = candidate.replace("\\", "/")
    candidate = _strip_repo_root(candidate, repo_root).replace("\\", "/")
    if candidate.startswith("/"):
        return None
    if candidate in {".", ".."}:
        return None
    while candidate.startswith("./"):
        candidate = candidate[2:]
    parts = [part for part in candidate.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        return None
    candidate = "/".join(parts)
    if " " in candidate or "\t" in candidate:
        return None
    # Avoid treating symbolic graph node ids or free-form labels as files.
    if "." not in parts[-1]:
        return None
    return candidate.lower()


def _collect_candidate_files_from_value(
    value: object,
    *,
    repo_root: str | None,
    trusted_file_context: bool,
) -> list[str]:
    candidates: list[str] = []
    if isinstance(value, str):
        if trusted_file_context:
            normalized = _normalize_candidate_file(value, repo_root)
            if normalized:
                candidates.append(normalized)
            return candidates
        try:
            parsed = json.loads(value)
        except Exception:
            return candidates
        return _collect_candidate_files_from_value(
            parsed,
            repo_root=repo_root,
            trusted_file_context=False,
        )
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_file_context = trusted_file_context or _is_file_key(key)
            candidates.extend(
                _collect_candidate_files_from_value(
                    child,
                    repo_root=repo_root,
                    trusted_file_context=child_file_context,
                )
            )
        return candidates
    if isinstance(value, list):
        for item in value:
            candidates.extend(
                _collect_candidate_files_from_value(
                    item,
                    repo_root=repo_root,
                    trusted_file_context=trusted_file_context,
                )
            )
        return candidates
    return candidates


def _collect_candidate_files_from_observations(
    observations: list[dict[str, object]],
    *,
    repo_root: str | None = None,
) -> list[str]:
    """Collect grounded repo-relative file candidates from structured observations."""
    seen: set[str] = set()
    candidates: list[str] = []
    for obs in reversed(observations):
        if not isinstance(obs, Mapping):
            continue
        for key in ("content", "result"):
            for candidate in _collect_candidate_files_from_value(
                obs.get(key),
                repo_root=repo_root,
                trusted_file_context=False,
            ):
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _extract_localization_prediction(
    agent_result: Mapping[str, object],
    *,
    repo_root: str | None = None,
) -> tuple[list[str], str | None, dict[str, object]]:
    """
    Emergency-only fallback: try to extract predicted_files from assistant content when finalization fails.
    This is a degraded path and should not be considered equivalent to schema-finalized output.
    """
    observations = agent_result.get("observations")
    if not isinstance(observations, list):
        return [], None, {"candidate_files": [], "reason": "missing_observations"}

    def _as_file_list(value: object) -> list[str]:
        cleaned = _collect_candidate_files_from_value(
            value,
            repo_root=repo_root,
            trusted_file_context=True,
        )
        # Preserve order while de-duplicating
        seen: set[str] = set()
        unique: list[str] = []
        for path in cleaned:
            if path in seen:
                continue
            seen.add(path)
            unique.append(path)
        return unique

    for obs in reversed(observations):
        if not isinstance(obs, Mapping):
            continue
        content = obs.get("content")
        reasoning: str | None = None
        if isinstance(content, Mapping):
            predicted = _as_file_list(content.get("predicted_files"))
            if not predicted:
                predicted = _as_file_list(content.get("files") or content.get("paths"))
            reason_val = content.get("reasoning")
            reasoning = reason_val if isinstance(reason_val, str) else None
            if predicted:
                return predicted, reasoning, {"candidate_files": predicted, "source": "assistant_content"}
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, Mapping):
                    predicted = _as_file_list(parsed.get("predicted_files"))
                    if not predicted:
                        predicted = _as_file_list(parsed.get("files") or parsed.get("paths"))
                    reason_val = parsed.get("reasoning")
                    reasoning = reason_val if isinstance(reason_val, str) else None
                    if predicted:
                        return predicted, reasoning, {"candidate_files": predicted, "source": "assistant_content_json"}
            except Exception:
                pass
    candidates = _collect_candidate_files_from_observations(
        [dict(obs) for obs in observations if isinstance(obs, Mapping)],
        repo_root=repo_root,
    )
    if candidates:
        return candidates, None, {"candidate_files": candidates, "source": "structured_observations"}
    return [], None, {"candidate_files": [], "source": "none"}


def _localization_result_schema() -> dict[str, object]:
    return {
        "name": "localization_result",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "predicted_files": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reasoning": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "file": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["file"],
                    },
                },
            },
            "required": ["predicted_files"],
        },
        "strict": True,
    }


def _format_finalization_messages(agent_task: dict[str, object], observations: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize exploration observations and request a structured result."""
    observation_snippets: list[str] = []
    for obs in observations[-5:]:
        try:
            observation_snippets.append(json.dumps(obs, default=str)[:400])
        except Exception:
            observation_snippets.append(str(obs)[:400])
    summary = "\n".join(observation_snippets)
    user_prompt = (
        "Based on the prior tool-assisted exploration, produce the final localization result.\n"
        "Return repo-relative `predicted_files` and brief reasoning.\n"
        "Observations:\n"
        f"{summary}"
    )
    return [
        {"role": "system", "content": "You are finalizing a bug localization task. Output must follow the provided JSON schema."},
        {"role": "user", "content": user_prompt},
    ]


def _finalize_localization(
    task: LCATask,
    observations: list[dict[str, object]],
    parent_trace: LangfuseSpan | LangfuseGeneration | None = None,
) -> LocalizationRunResult:
    client, model = _make_client()
    agent_task = serialize_lca_task_for_prompt(task)
    repo_value = agent_task.get("repo") if isinstance(agent_task, Mapping) else None
    repo_root = repo_value if isinstance(repo_value, str) else None
    prepared_observations = _clean_observations_for_finalization(observations, repo_root)
    messages = _format_finalization_messages(agent_task, prepared_observations)
    schema = _localization_result_schema()
    parent_span = _require_parent(parent_trace, owner="runner agent")
    with parent_span.start_as_current_observation(
        name="localization_finalize",
        as_type="span",
        metadata=serialize_observation_metadata(
            {"phase": "finalization", "runner_model": model}
        ),
    ) as obs:
        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": schema},
            extra_headers={"x-cerebras-api-key": get_cerebras_api_key()},
        )
        usage_details = _collect_usage_details(completion)
        if usage_details:
            obs.update(usage_details=_langfuse_usage_details(usage_details))
        choice = completion.choices[0]
        message_obj = getattr(choice, "message", None)
        content = getattr(message_obj, "content", None) if message_obj is not None else None
        parsed: dict[str, object] = {}
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:  # noqa: BLE001
                raise RuntimeError(f"Invalid structured output: {exc}") from exc
        elif isinstance(content, Mapping):
            parsed = dict(content)
        else:
            raise RuntimeError("Structured output missing content")
        obs.update(output=_truncate_for_metadata(parsed))
        result = LocalizationRunResult.model_validate(
            {
                "task": task,
                "prediction": _prediction_from_output(
                    parsed.get("predicted_files", []),
                    parsed.get("reasoning"),
                ),
                "observations": prepared_observations or observations,
                "raw": {
                    "final": parsed,
                    "usage_details": usage_details,
                    "generation_count": 1,
                },
                "source": "finalize",
            }
        )
        return result


def _fallback_localization_result(
    *,
    task: LCATask,
    normalized: Mapping[str, object],
    observations: list[dict[str, object]],
    node_count: int | None,
    repo_root: str | None,
    error: object,
) -> LocalizationRunResult:
    predicted_files, reasoning, diagnostics = _extract_localization_prediction(
        normalized,
        repo_root=repo_root,
    )
    return LocalizationRunResult(
        task=task,
        prediction=_prediction_from_output(predicted_files, reasoning),
        observations=observations,
        node_count=node_count,
        raw={
            "error": str(error),
            "fallback_raw": normalized.get("raw"),
            "fallback_diagnostics": diagnostics,
        },
        source="fallback" if predicted_files else "fallback_empty",
    )


def _finalize_or_recover(
    *,
    task: LCATask,
    observations: list[dict[str, object]],
    normalized: Mapping[str, object],
    node_count: int | None,
    parent_span: LangfuseSpan | LangfuseGeneration,
) -> LocalizationRunResult:
    try:
        final = _finalize_localization(task, observations, parent_trace=parent_span)
    except Exception as exc:  # noqa: BLE001
        agent_task = serialize_lca_task_for_prompt(task)
        repo_value = agent_task.get("repo")
        return _fallback_localization_result(
            task=task,
            normalized=normalized,
            observations=observations,
            node_count=node_count,
            repo_root=repo_value if isinstance(repo_value, str) else None,
            error=exc,
        )
    if final.prediction.predicted_files:
        return final
    agent_task = serialize_lca_task_for_prompt(task)
    repo_value = agent_task.get("repo")
    return _fallback_localization_result(
        task=task,
        normalized=normalized,
        observations=observations,
        node_count=node_count,
        repo_root=repo_value if isinstance(repo_value, str) else None,
        error="finalization returned empty predicted_files",
    )


def _runner_error_result(
    *,
    task: LCATask,
    normalized: Mapping[str, object],
    observations: list[dict[str, object]],
    node_count: int | None,
) -> LocalizationRunResult | None:
    error = normalized.get("error")
    if not error or observations:
        return None
    return LocalizationRunResult(
        task=task,
        prediction=LocalizationPrediction(predicted_files=[]),
        observations=[],
        node_count=node_count,
        raw={
            "error": str(error),
            "fallback_diagnostics": {
                "candidate_files": [],
                "source": "runner_error_no_observations",
            },
        },
        source="runner_error",
    )


def run_ic_iteration(
    task: LCATask,
    score_fn: Callable[..., float] | None = None,
    steps: int = 5,
    parent_trace: LangfuseSpan | LangfuseGeneration | None = None,
):
    started_at = time.perf_counter()
    parent_span = _require_parent(parent_trace, owner="iterative_context")
    if not task.repo:
        raise ValueError("LCATask.repo is required for iterative_context execution")
    repo = task.repo
    agent_task = serialize_lca_task_for_prompt(task)

    with parent_span.start_as_current_observation(
        name="iterative_context",
        as_type="span",
        metadata=serialize_observation_metadata(
            {
                "backend": "iterative_context",
                "repo": repo,
                "identity": task.identity.task_id(),
                "phase": "backend_init",
            }
        ),
    ) as backend_obs:
        try:
            backend = IterativeContextBackend(repo=repo, score_fn=score_fn)
            tool_specs: list[ChatCompletionToolParam] = backend.tool_specs
            system_prompt = build_ic_system_prompt(tool_specs)
            try:
                backend_obs.update(
                    metadata=serialize_observation_metadata(
                        {
                            "backend": "iterative_context",
                            "repo": repo,
                            "identity": task.identity.task_id(),
                            "phase": "agent_execution",
                            "tool_count": len(tool_specs),
                        }
                    )
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "langfuse_metadata_update_failed",
                    operation="iterative_context.update",
                    observation=_span_label(backend_obs),
                    keys=["backend", "identity", "phase", "repo", "tool_count"],
                    error=str(exc),
                )
        except Exception as exc:  # noqa: BLE001
            summary = _build_task_run_summary(
                task=task,
                backend="ic",
                status="failure",
                started_at=started_at,
                raw={},
                observations=[],
                stage="backend_init",
                message=str(exc),
            )
            try:
                backend_obs.update(
                    metadata=serialize_observation_metadata(
                        {
                            "task_summary": summary.model_dump(mode="json"),
                            "phase": "backend_init",
                            "error": str(exc),
                        }
                    )
                )
            except Exception as update_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "langfuse_metadata_update_failed",
                    operation="iterative_context.update",
                    observation=_span_label(backend_obs),
                    keys=["error", "phase", "task_summary"],
                    error=str(update_exc),
                )
            raise LocalizationEvaluationError(
                LocalizationFailureCategory.RUNNER,
                str(exc),
                task_id=task.task_id,
                observability_summary=summary,
            ) from exc
        try:
            raw = run_agent(
                agent_task,
                steps=steps,
                tool_specs=tool_specs,
                dispatch_tool_call=backend.dispatch,
                system_prompt=system_prompt,
                parent_trace=backend_obs,
                backend_name="iterative_context",
            )
        except Exception as exc:  # noqa: BLE001
            raw = cast(dict[str, object], {"observations": [], "error": str(exc)})
            try:
                backend_obs.update(
                    metadata=serialize_observation_metadata(
                        {"error": str(exc), "phase": "agent_execution"}
                    )
                )
            except Exception as update_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "langfuse_metadata_update_failed",
                    operation="iterative_context.update",
                    observation=_span_label(backend_obs),
                    keys=["error", "phase"],
                    error=str(update_exc),
                )
        normalized = _normalize_agent_result(raw)
        observations = _coerce_observations(normalized.get("observations"))
        normalized_node_count = _coerce_node_count(normalized.get("node_count"))
        node_count_val = normalized_node_count
        # If backend didn't supply node_count, fall back to full_graph nodes length.
        if node_count_val is None:
            full_graph = normalized.get("full_graph")
            if isinstance(full_graph, Mapping):
                nodes = full_graph.get("nodes")
                if isinstance(nodes, list):
                    node_count_val = len(nodes)
        # If still missing, look into latest observation result full_graph.
        if node_count_val is None:
            for latest in reversed(observations):
                if not isinstance(latest, Mapping):
                    continue
                result = latest.get("result")
                if not isinstance(result, Mapping):
                    continue
                full_graph = result.get("full_graph")
                if isinstance(full_graph, Mapping):
                    nodes = full_graph.get("nodes")
                    if isinstance(nodes, list):
                        node_count_val = len(nodes)
                        break
        node_count_metric = float(node_count_val) if node_count_val is not None else 0.0
        try:
            backend_obs.score(
                name="ic.node_count",
                value=node_count_metric,
                data_type=ScoreDataType.NUMERIC,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "langfuse_score_emission_failed",
                operation="iterative_context.score",
                observation=_span_label(backend_obs),
                score="ic.node_count",
                error=str(exc),
            )
        try:
            backend_obs.update(
                metadata=serialize_observation_metadata(
                    {"node_count": node_count_val or 0, "full_graph": normalized.get("full_graph")}
                )
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "langfuse_metadata_update_failed",
                operation="iterative_context.update",
                observation=_span_label(backend_obs),
                keys=["full_graph", "node_count"],
                error=str(exc),
            )

    final = _runner_error_result(
        task=task,
        normalized=normalized,
        observations=observations,
        node_count=node_count_val if node_count_val is not None else normalized_node_count,
    )
    if final is None:
        final = _finalize_or_recover(
            task=task,
            observations=observations,
            normalized=normalized,
            node_count=node_count_val if node_count_val is not None else normalized_node_count,
            parent_span=parent_span,
        )
    if not final.observations:
        final.observations = observations
    if final.node_count is None:
        final.node_count = node_count_val if node_count_val is not None else normalized_node_count
    full_graph = normalized.get("full_graph")
    if full_graph is not None:
        final.full_graph = full_graph
        if final.node_count in (None, 0) and isinstance(full_graph, Mapping):
            nodes = full_graph.get("nodes")
            if isinstance(nodes, list):
                final.node_count = len(nodes)
    final.backend = "ic"
    _preserve_agent_raw(final, raw)
    _attach_task_summary(
        final,
        task=task,
        backend="ic",
        started_at=started_at,
        stage="agent_execution" if final.source == "runner_error" else ("finalization" if not final.prediction.predicted_files else None),
        message=str((final.raw or {}).get("error")) if isinstance(final.raw, Mapping) and (final.raw or {}).get("error") else None,
    )
    try:
        backend_obs.update(
            metadata=serialize_observation_metadata(
                {
                    "task_summary": final.observability_summary.model_dump(mode="json")
                    if final.observability_summary
                    else None
                }
            )
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "langfuse_metadata_update_failed",
            operation="iterative_context.update",
            observation=_span_label(backend_obs),
            keys=["task_summary"],
            error=str(exc),
        )
    return final


def run_jc_iteration(
    task: LCATask,
    steps: int = 5,
    parent_trace: LangfuseSpan | LangfuseGeneration | None = None,
):
    started_at = time.perf_counter()
    parent_span = _require_parent(parent_trace, owner="jcodemunch")
    if not task.repo:
        raise ValueError("LCATask.repo is required for jcodemunch execution")
    repo = task.repo
    agent_task = serialize_lca_task_for_prompt(task)
    backend_logger = bind_logger(
        _LOGGER,
        backend="jcodemunch",
        task=short_task_label(task),
        task_identity=task.task_id,
        repo=repo,
    )

    with parent_span.start_as_current_observation(
        name="jcodemunch",
        as_type="span",
        metadata=serialize_observation_metadata(
            {
                "backend": "jcodemunch",
                "repo": repo,
                "identity": task.identity.task_id(),
                "phase": "backend_init",
            }
        ),
    ) as backend_obs:
        backend_logger.info("backend_init_started")
        try:
            backend = JCodeMunchBackend(repo=repo)
            tool_specs: list[ChatCompletionToolParam] = backend.tool_specs
            system_prompt = build_jc_system_prompt(tool_specs)
        except Exception as exc:  # noqa: BLE001
            summary = _build_task_run_summary(
                task=task,
                backend="jc",
                status="failure",
                started_at=started_at,
                raw={},
                observations=[],
                stage="backend_init",
                message=str(exc),
            )
            backend_logger.error(
                "backend_init_failed",
                error=str(exc),
            )
            try:
                backend_obs.update(
                    metadata=serialize_observation_metadata(
                        {
                            "backend": "jcodemunch",
                            "repo": repo,
                            "identity": task.identity.task_id(),
                            "phase": "backend_init",
                            "error": str(exc),
                            "task_summary": summary.model_dump(mode="json"),
                        }
                    )
                )
            except Exception as update_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "langfuse_metadata_update_failed",
                    operation="jcodemunch.update",
                    observation=_span_label(backend_obs),
                    keys=["backend", "error", "identity", "phase", "repo", "task_summary"],
                    error=str(update_exc),
                )
            raise LocalizationEvaluationError(
                LocalizationFailureCategory.RUNNER,
                str(exc),
                task_id=task.task_id,
                observability_summary=summary,
            ) from exc
        backend_logger.info(
            "backend_init_completed",
            tool_count=len(tool_specs),
        )
        try:
            backend_obs.update(
                metadata=serialize_observation_metadata(
                    {
                        "backend": "jcodemunch",
                        "repo": repo,
                        "identity": task.identity.task_id(),
                        "phase": "agent_execution",
                        "tool_count": len(tool_specs),
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "langfuse_metadata_update_failed",
                operation="jcodemunch.update",
                observation=_span_label(backend_obs),
                keys=["backend", "identity", "phase", "repo", "tool_count"],
                error=str(exc),
            )
        try:
            raw = run_agent(
                agent_task,
                steps=steps,
                tool_specs=tool_specs,
                dispatch_tool_call=backend.dispatch,
                system_prompt=system_prompt,
                parent_trace=backend_obs,
                backend_name="jcodemunch",
            )
        except Exception as exc:  # noqa: BLE001
            raw = cast(dict[str, object], {"observations": [], "error": str(exc)})
            try:
                backend_obs.update(
                    metadata=serialize_observation_metadata(
                        {"error": str(exc), "phase": "agent_execution"}
                    )
                )
            except Exception as update_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "langfuse_metadata_update_failed",
                    operation="jcodemunch.update",
                    observation=_span_label(backend_obs),
                    keys=["error", "phase"],
                    error=str(update_exc),
                )
        normalized = _normalize_agent_result(raw)
        observations = _coerce_observations(normalized.get("observations"))
        node_count_val = _coerce_node_count(normalized.get("node_count"))
        if node_count_val is None:
            full_graph = normalized.get("full_graph")
            if isinstance(full_graph, Mapping):
                nodes = full_graph.get("nodes")
                if isinstance(nodes, list):
                    node_count_val = len(nodes)
        node_count_metric = float(node_count_val) if node_count_val is not None else 0.0
        try:
            backend_obs.score(
                name="jc.node_count",
                value=node_count_metric,
                data_type=ScoreDataType.NUMERIC,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "langfuse_score_emission_failed",
                operation="jcodemunch.score",
                observation=_span_label(backend_obs),
                score="jc.node_count",
                error=str(exc),
            )
        try:
            backend_obs.update(
                metadata=serialize_observation_metadata({"node_count": node_count_val or 0})
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "langfuse_metadata_update_failed",
                operation="jcodemunch.update",
                observation=_span_label(backend_obs),
                keys=["node_count"],
                error=str(exc),
            )

    final = _runner_error_result(
        task=task,
        normalized=normalized,
        observations=observations,
        node_count=node_count_val,
    )
    if final is None:
        final = _finalize_or_recover(
            task=task,
            observations=observations,
            normalized=normalized,
            node_count=node_count_val,
            parent_span=parent_span,
        )
    if not final.observations:
        final.observations = observations
    if final.node_count is None:
        final.node_count = node_count_val
    final.backend = "jc"
    _preserve_agent_raw(final, raw)
    _attach_task_summary(
        final,
        task=task,
        backend="jc",
        started_at=started_at,
        stage="agent_execution" if final.source == "runner_error" else ("finalization" if not final.prediction.predicted_files else None),
        message=str((final.raw or {}).get("error")) if isinstance(final.raw, Mapping) and (final.raw or {}).get("error") else None,
    )
    try:
        backend_obs.update(
            metadata=serialize_observation_metadata(
                {
                    "task_summary": final.observability_summary.model_dump(mode="json")
                    if final.observability_summary
                    else None
                }
            )
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "langfuse_metadata_update_failed",
            operation="jcodemunch.update",
            observation=_span_label(backend_obs),
            keys=["task_summary"],
            error=str(exc),
        )
    return final


__all__ = [
    "run_ic_iteration",
    "run_jc_iteration",
    "run_agent",
    "run_agent_with_retry",
    "LocalizationPrediction",
    "LocalizationRunResult",
]
