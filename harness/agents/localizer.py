from __future__ import annotations

"""
Localizer agent: LLM/tool-execution layer for iterative-context and jcodemunch
bug-localization flows.  Owns message construction, model calls, tool dispatch,
context-budget management, finalization, and retry logic.
"""

import json
import logging
from collections.abc import Mapping
from typing import Any, Callable, cast

from harness.telemetry.tracing import (
    UsageDetails,
    get_tracing_openai_client,
    start_observation,
)
from harness.telemetry.tracing.score_emitter import emit_score_for_handle
from harness.prompts import SystemPromptContext
from harness.backends.ic import IterativeContextBackend
from harness.backends.jc import JCodeMunchBackend
from harness.backends.mcp import serialize_tool_result_for_model
from harness.utils.env import get_cerebras_api_key, get_runner_model, getenv
from harness.utils.openai_schema import ChatCompletionToolParam, validate_tools
from harness.utils.template_loader import render_prompt_template
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from harness.localization.models import LCAContext, LCATaskIdentity

from harness.agents.common import AgentTaskPayload, usage_from_response

_MAX_TOTAL_MESSAGE_CHARS = 24000
_MAX_TOOL_CONTENT_CHARS = 4000
_MAX_SYSTEM_CHARS = 4000
_MAX_USER_CHARS = 4000
_HISTORY_LIMIT = 3
_AGGRESSIVE_TOTAL_CHARS = 12000
_AGGRESSIVE_TOOL_CONTENT_CHARS = 1200
_AGGRESSIVE_SYSTEM_CHARS = 1500
_AGGRESSIVE_USER_CHARS = 1500

_LOGGER = logging.getLogger(__name__)


class AgentTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity: LCATaskIdentity
    repo: str
    context: LCAContext
    extra: dict[str, object] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        payload_model = AgentTaskPayload(
            identity=self.identity.model_dump(),
            repo=self.repo,
            context=self.context.model_dump(),
            extra=self.extra,
        )
        return payload_model.model_dump()


class LocalizationRunnerResult(BaseModel):
    """Typed localization runner output boundary."""

    model_config = ConfigDict(extra="forbid")

    predicted_files: list[str] = Field(default_factory=list)
    reasoning: str | None = None
    observations: list[dict[str, object]] = Field(default_factory=list)
    node_count: int | None = None
    raw: dict[str, object] | None = None
    source: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        if isinstance(value, Mapping):
            files_raw = value.get("predicted_files", [])
            normalized: list[str] = []
            seen: set[str] = set()
            if isinstance(files_raw, list):
                for item in files_raw:
                    if not isinstance(item, str):
                        continue
                    cleaned = item.strip().lower()
                    if cleaned and cleaned not in seen:
                        seen.add(cleaned)
                        normalized.append(cleaned)
            elif isinstance(files_raw, str):
                for part in files_raw.replace("\n", ",").split(","):
                    cleaned = part.strip().lower()
                    if cleaned and cleaned not in seen:
                        seen.add(cleaned)
                        normalized.append(cleaned)
            return {**value, "predicted_files": normalized}
        return value


def _require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"task must include a non-empty string for '{key}'")
    return value


def _require_parent(parent: object | None, owner: str = "runner agent") -> object:
    if parent is None:
        raise RuntimeError(f"Parent span required for {owner} tracing")
    return parent


def _safe_update(span: object | None, **kwargs: object) -> None:
    updater = getattr(span, "update", None)
    if callable(updater):
        try:
            updater(**kwargs)
        except Exception:
            pass


def _tool_names_from_specs(tool_specs: list[ChatCompletionToolParam]) -> list[str]:
    names: list[str] = []
    for spec in tool_specs:
        func = spec.get("function") if isinstance(spec, Mapping) else None  # pyright: ignore[reportUnnecessaryIsInstance]
        name = func.get("name") if isinstance(func, Mapping) else None  # pyright: ignore[reportUnnecessaryIsInstance]
        if isinstance(name, str):
            names.append(name)
    return names


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


def _format_available_tools(tool_names: list[str]) -> str:
    if not tool_names:
        return "- none"
    return "\n".join(f"- {name}" for name in tool_names)


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
    _LOGGER.info("Localization runner model resolved: %s (override=%s, env=%s)", resolved, bool(model_override), env_model)
    return resolved


def _make_client(model_override: str | None = None) -> tuple[Any, str]:
    api_key = get_cerebras_api_key()
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is required for runner agent client")
    model_name = resolve_runner_model(model_override)
    return get_tracing_openai_client(base_url="https://api.cerebras.ai/v1", api_key=api_key), model_name


def _build_ic_system_prompt(tool_specs: list[ChatCompletionToolParam], aggressive: bool = False) -> str:
    tool_names = _tool_names_from_specs(tool_specs)
    ctx = SystemPromptContext(available_tools=_format_available_tools(tool_names))
    return render_prompt_template("ic_system.jinja", ctx.model_dump())


def _build_jc_system_prompt(tool_specs: list[ChatCompletionToolParam]) -> str:
    tool_names = _tool_names_from_specs(tool_specs)
    ctx = SystemPromptContext(available_tools=_format_available_tools(tool_names))
    return render_prompt_template("jc_system.jinja", ctx.model_dump())


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
    parent_trace: object | None = None,
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
            with start_observation(name=f"toolcall.{name}", parent=parent_span, metadata={"tool": name}) as tool_obs:
                result = dispatch_tool_call(name, arg_map)
                new_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_map.get("id"),
                        "content": _compact_tool_content(result),
                        "result": result,
                    }
                )
                tool_obs.update(output=_truncate_for_metadata(result))
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
    new_obs: list[dict[str, object]] = []
    tool_only: list[dict[str, object]] = []
    for obs in observations:
        new_obs.append(obs)
        if obs.get("role") == "tool":
            tool_only.append(obs)
        if obs.get("role") == "assistant" and obs.get("tool_calls"):
            new_obs.extend(tool_results)
            tool_only.extend(tool_results)
    ordered_obs = [obs for obs in new_obs if obs.get("role") != "tool"] + tool_only
    return _enforce_message_budget(
        _build_model_messages(agent_task, tool_specs, system_prompt) + ordered_obs,
        max_total_chars=_AGGRESSIVE_TOTAL_CHARS,
        max_tool_content=_AGGRESSIVE_TOOL_CONTENT_CHARS,
        max_system_chars=_AGGRESSIVE_SYSTEM_CHARS,
        max_user_chars=_AGGRESSIVE_USER_CHARS,
        keep_last_n_tools=max(1, len(tool_only)),
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


def run_agent(
    agent_task: dict[str, object],
    steps: int,
    tool_specs: list[ChatCompletionToolParam],
    dispatch_tool_call: Callable[[str, dict[str, Any]], object],
    system_prompt: str,
    parent_trace: object | None = None,
    backend_name: str | None = None,
):
    parent_span = _require_parent(parent_trace, owner=backend_name or "runner agent")
    client, model = _make_client()
    _safe_update(parent_span, metadata={"runner_model": model})
    messages = _build_model_messages(agent_task, tool_specs, system_prompt)
    tool_choice = "required" if tool_specs else None
    tools_payload: list[ChatCompletionToolParam] | None = tool_specs or None
    observations: list[dict[str, object]] = []

    for step in range(steps):
        with start_observation(
            name="agent_step",
            parent=parent_span,
            metadata={"step": step, "model": model, "backend": backend_name or "unknown"},
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
                obs.update(usage_details=usage_details)
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
                tool_results = _collect_tool_results(
                    observations=[{"tool_call": call} for call in tool_calls],
                    dispatch_tool_call=dispatch_tool_call,
                    budget_limit=_MAX_TOTAL_MESSAGE_CHARS,
                    parent_trace=obs,
                )
                if tool_results:
                    observations.extend(tool_results)
                observations.append(
                    {
                        "role": "assistant",
                        "tool_calls": tool_calls,
                        "content": response_message.get("content") or "",
                    }
                )
                messages = _apply_tool_results(agent_task, observations, tool_results, system_prompt, tool_specs)
                continue
            else:
                raw_content = response_message.get("content")
                content_text = raw_content if isinstance(raw_content, str) else "{}"
                try:
                    response_json = json.loads(content_text)
                except json.JSONDecodeError as exc:  # noqa: BLE001
                    response_json = {"error": f"invalid json: {exc}"}
                obs.update(output=_truncate_for_metadata(response_json))
                observations.append({"role": "assistant", "content": response_json})
                break
    return {"observations": observations}


def run_agent_with_retry(
    agent_task: dict[str, object],
    steps: int,
    tool_specs: list[ChatCompletionToolParam],
    dispatch_tool_call: Callable[[str, dict[str, Any]], object],
    system_prompt: str,
    parent_trace: object | None = None,
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


def _extract_localization_prediction(agent_result: Mapping[str, object]) -> tuple[list[str], str | None]:
    """
    Emergency-only fallback: try to extract predicted_files from assistant content when finalization fails.
    This is a degraded path and should not be considered equivalent to schema-finalized output.
    """
    observations = agent_result.get("observations")
    if not isinstance(observations, list):
        return [], None

    def _as_file_list(value: object) -> list[str]:
        # Fallback only accepts already-structured array or JSON string.
        cleaned: list[str] = []
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, str):
                    stripped = entry.strip()
                    if stripped:
                        cleaned.append(stripped.lower())
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    for entry in parsed:
                        if isinstance(entry, str):
                            stripped = entry.strip()
                            if stripped:
                                cleaned.append(stripped.lower())
            except Exception:
                return []
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
                return predicted, reasoning
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, Mapping):
                    predicted = _as_file_list(parsed.get("predicted_files"))
                    reason_val = parsed.get("reasoning")
                    reasoning = reason_val if isinstance(reason_val, str) else None
                    if predicted:
                        return predicted, reasoning
            except Exception:
                return [], None
    return [], None


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
                    "default": [],
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
                    "default": [],
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
    agent_task: dict[str, object],
    observations: list[dict[str, object]],
    parent_trace: object | None = None,
) -> LocalizationRunnerResult:
    client, model = _make_client()
    repo_value = agent_task.get("repo") if isinstance(agent_task, Mapping) else None
    repo_root = repo_value if isinstance(repo_value, str) else None
    prepared_observations = _clean_observations_for_finalization(observations, repo_root)
    messages = _format_finalization_messages(agent_task, prepared_observations)
    schema = _localization_result_schema()
    with start_observation(
        name="localization_finalize",
        parent=_require_parent(parent_trace, owner="runner agent"),
        metadata={"phase": "finalize", "runner_model": model},
    ) as obs:
        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": schema},
            extra_headers={"x-cerebras-api-key": get_cerebras_api_key()},
        )
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
        result = LocalizationRunnerResult.model_validate(
            {
                "predicted_files": parsed.get("predicted_files", []),
                "reasoning": parsed.get("reasoning"),
                "observations": prepared_observations or observations,
                "raw": {"final": parsed},
                "source": "finalize",
            }
        )
        return result


def run_ic_iteration(
    task: dict[str, object],
    score_fn: Callable[..., float] | None = None,
    steps: int = 5,
    parent_trace: object | None = None,
):
    parent_span = _require_parent(parent_trace, owner="iterative_context")
    repo = _require_str(task, "repo")
    identity = LCATaskIdentity.model_validate(task.get("identity") or {})
    context = LCAContext.model_validate(task.get("context") or {})
    try:
        agent_task_model = AgentTask(
            identity=identity,
            repo=repo,
            context=context,
            extra={k: v for k, v in task.items() if k not in {"identity", "context", "repo", "changed_files"}},
        )
    except ValidationError as exc:
        raise RuntimeError(f"Invalid agent task: {exc}") from exc
    agent_task = agent_task_model.to_payload()

    backend = IterativeContextBackend(repo=repo, score_fn=score_fn)
    tool_specs: list[ChatCompletionToolParam] = backend.tool_specs
    system_prompt = _build_ic_system_prompt(tool_specs)
    with start_observation(
        name="iterative_context",
        parent=parent_span,
        metadata={"backend": "iterative_context", "repo": repo, "identity": identity.task_id()},
    ) as backend_obs:
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
                backend_obs.update(metadata={"error": str(exc)})
            except Exception:
                pass
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
        emit_score_for_handle(backend_obs, name="ic.node_count", value=node_count_metric, data_type="NUMERIC")
        backend_obs.update(metadata={"node_count": node_count_val or 0, "full_graph": normalized.get("full_graph")})

    try:
        final = _finalize_localization(agent_task, observations, parent_trace=parent_span)
    except (json.JSONDecodeError, ValidationError) as exc:
        predicted_files, reasoning = _extract_localization_prediction(normalized)
        final = LocalizationRunnerResult(
            predicted_files=predicted_files,
            reasoning=reasoning,
            observations=observations,
            node_count=node_count_val if node_count_val is not None else normalized_node_count,
            raw={"error": str(exc), "fallback_raw": normalized.get("raw")},
            source="fallback",
        )
    if not final.observations:
        final.observations = observations
    if final.node_count is None:
        final.node_count = node_count_val if node_count_val is not None else normalized_node_count
    result_dict = final.model_dump()
    full_graph = normalized.get("full_graph")
    if full_graph is not None:
        result_dict["full_graph"] = full_graph
        if result_dict.get("node_count") in (None, 0) and isinstance(full_graph, Mapping):
            nodes = full_graph.get("nodes")
            if isinstance(nodes, list):
                result_dict["node_count"] = len(nodes)
    return result_dict


def run_jc_iteration(
    task: dict[str, object],
    steps: int = 5,
    parent_trace: object | None = None,
):
    parent_span = _require_parent(parent_trace, owner="jcodemunch")
    repo = _require_str(task, "repo")
    identity = LCATaskIdentity.model_validate(task.get("identity") or {})
    context = LCAContext.model_validate(task.get("context") or {})
    try:
        agent_task_model = AgentTask(
            identity=identity,
            repo=repo,
            context=context,
            extra={k: v for k, v in task.items() if k not in {"identity", "context", "repo", "changed_files"}},
        )
    except ValidationError as exc:
        raise RuntimeError(f"Invalid agent task: {exc}") from exc
    agent_task = agent_task_model.to_payload()

    backend = JCodeMunchBackend(repo=repo)
    tool_specs: list[ChatCompletionToolParam] = backend.tool_specs
    system_prompt = _build_jc_system_prompt(tool_specs)
    with start_observation(
        name="jcodemunch",
        parent=parent_span,
        metadata={"backend": "jcodemunch", "repo": repo, "identity": identity.task_id()},
    ) as backend_obs:
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
                backend_obs.update(metadata={"error": str(exc)})
            except Exception:
                pass
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
        emit_score_for_handle(backend_obs, name="jc.node_count", value=node_count_metric, data_type="NUMERIC")
        backend_obs.update(metadata={"node_count": node_count_val or 0})

    try:
        final = _finalize_localization(agent_task, observations, parent_trace=parent_span)
    except (json.JSONDecodeError, ValidationError) as exc:
        predicted_files, reasoning = _extract_localization_prediction(normalized)
        final = LocalizationRunnerResult(
            predicted_files=predicted_files,
            reasoning=reasoning,
            observations=observations,
            node_count=node_count_val,
            raw={"error": str(exc), "fallback_raw": normalized.get("raw")},
            source="fallback",
        )
    if not final.observations:
        final.observations = observations
    if final.node_count is None:
        final.node_count = node_count_val
    return final.model_dump()


__all__ = [
    "run_ic_iteration",
    "run_jc_iteration",
    "run_agent",
    "run_agent_with_retry",
    "_build_ic_system_prompt",
    "_build_jc_system_prompt",
]
