from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Callable, cast

from .observability.langfuse import start_span, get_tracing_openai_client
from .observability.score_emitter import emit_score_for_handle
from .tools.backends.ic_backend import IterativeContextBackend
from .tools.backends.jc_backend import JCodeMunchBackend
from .tools.mcp_adapter import serialize_tool_result_for_model
from .utils.env import get_cerebras_api_key, get_runner_model
from .utils.openai_schema import OpenAITool, build_tool_response, validate_tools
from .prompts import SystemPromptContext
from .utils.template_loader import render_prompt_template

_MAX_TOTAL_MESSAGE_CHARS = 24000
_MAX_TOOL_CONTENT_CHARS = 4000
_MAX_SYSTEM_CHARS = 4000
_MAX_USER_CHARS = 4000
_HISTORY_LIMIT = 3
_AGGRESSIVE_TOTAL_CHARS = 12000
_AGGRESSIVE_TOOL_CONTENT_CHARS = 1200
_AGGRESSIVE_SYSTEM_CHARS = 1500
_AGGRESSIVE_USER_CHARS = 1500


def _require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"task must include a non-empty string for '{key}'")
    return value


def _tool_names_from_specs(tool_specs: list[OpenAITool]) -> list[str]:
    names: list[str] = []
    for spec in tool_specs:
        func = spec.get("function") if isinstance(spec, Mapping) else None  # pyright: ignore[reportUnnecessaryIsInstance]
        name = func.get("name") if isinstance(func, Mapping) else None  # pyright: ignore[reportUnnecessaryIsInstance]
        if isinstance(name, str):
            names.append(name)
    return names


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
    for idx, msg in enumerate(messages):
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
    except Exception:
        serialized = str(payload)
    if len(serialized) <= limit:
        return payload
    return f"{serialized[:limit]}...(truncated)"


def _build_ic_system_prompt(tool_specs: list[OpenAITool]) -> str:
    tool_names = _tool_names_from_specs(tool_specs)
    context = SystemPromptContext(available_tools=_format_available_tools(tool_names))
    return render_prompt_template("ic_system.jinja", context.model_dump())


def _build_jc_system_prompt(tool_specs: list[OpenAITool]) -> str:
    tool_names = _tool_names_from_specs(tool_specs)
    tools_section = _format_available_tools(tool_names)
    context = SystemPromptContext(available_tools=tools_section)
    return render_prompt_template("jc_system.jinja", context.model_dump())


def _make_client():
    api_key = get_cerebras_api_key()
    if not api_key:
        raise RuntimeError("Missing CEREBRAS_API_KEY in environment")
    model = get_runner_model()
    client = get_tracing_openai_client(base_url="https://api.cerebras.ai/v1", api_key=api_key)
    return client, model


def run_agent(
    task: dict[str, object],
    steps: int,
    tool_specs: list[OpenAITool],
    dispatch_tool_call: Callable[[str, dict[str, Any]], Any],
    system_prompt: str,
    parent_trace: object | None = None,
    backend_name: str | None = None,
    model: str | None = None,
) -> dict[str, object]:
    """
    Execute a deterministic tool-only agent for a fixed number of steps.
    """
    client, resolved_model = _make_client()
    model_name = model or resolved_model
    validate_tools(tool_specs)

    messages: list[Any] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Task:\n{task}\n\nUse tools to explore.",
        },
    ]

    observations: list[dict[str, object]] = []
    agent_obs = start_span(parent_trace, "agent_run", metadata={"backend": backend_name or "unknown", "model": model_name, "task": task})
    for step_index in range(steps):
        model_obs = start_span(agent_obs, "model_completion", metadata={"model": model_name, "backend": backend_name, "step": step_index})
        start_time = time.perf_counter()
        response = None
        attempt = 0
        while True:
            try:
                budgeted = _aggressive_compact_messages(messages) if attempt else _enforce_message_budget(messages)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=cast(Any, budgeted),
                    tools=cast(Any, tool_specs),
                    tool_choice="required",
                )
                break
            except Exception as exc:  # noqa: BLE001
                if not _is_context_error(exc) or attempt:
                    if model_obs and hasattr(model_obs, "end") and response is None:
                        try:
                            model_obs.end(error=str(exc))
                        except Exception:
                            pass
                    raise
                attempt += 1
                continue
        latency_ms = (time.perf_counter() - start_time) * 1000
        if model_obs and hasattr(model_obs, "end"):
            try:
                model_obs.end(metadata={"latency_ms": latency_ms})
            except Exception:
                pass

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            if agent_obs and hasattr(agent_obs, "end"):
                try:
                    agent_obs.end(error="missing_tool_call")
                except Exception:
                    pass
            raise RuntimeError("Model failed to produce tool call")

        call = tool_calls[0]
        call_id = getattr(call, "id", None)
        if not call_id:
            if agent_obs and hasattr(agent_obs, "end"):
                try:
                    agent_obs.end(error="missing_call_id")
                except Exception:
                    pass
            raise RuntimeError("Tool call missing id")
        try:
            raw_args = call.function.arguments or {}
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            tool_name = call.function.name
            tool_obs = start_span(agent_obs, tool_name or "tool_call", metadata={"backend": backend_name, "step": step_index, "tool": tool_name, "args": _truncate_for_metadata(parsed_args)})
            tool_start = time.perf_counter()
            result = dispatch_tool_call(tool_name, parsed_args)
            latency_ms = (time.perf_counter() - tool_start) * 1000
            observations.append({"tool": tool_name, "result": result})
            messages.append(build_tool_response(call_id, _compact_tool_content(result)))
            if tool_obs and hasattr(tool_obs, "end"):
                try:
                    tool_obs.end(output={"result": _truncate_for_metadata(result)}, metadata={"latency_ms": latency_ms, "tool": tool_name, "step": step_index})
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            tool_name = call.function.name or "unknown"
            observations.append({"tool": tool_name, "error": str(exc)})
            messages.append(build_tool_response(call_id, f"ERROR: {exc}"))
            err_obs = start_span(agent_obs, f"{tool_name}_error", metadata={"backend": backend_name, "step": step_index, "tool": tool_name, "error": str(exc)})
            if err_obs and hasattr(err_obs, "end"):
                try:
                    err_obs.end(error=exc)
                except Exception:
                    pass
        messages = _enforce_message_budget(messages)

    if not observations:
        if agent_obs and hasattr(agent_obs, "end"):
            try:
                agent_obs.end(error="no_observations")
            except Exception:
                pass
        raise RuntimeError("Agent failed to call any tools")

    if agent_obs and hasattr(agent_obs, "end"):
        try:
            agent_obs.end(metadata={"backend": backend_name, "model": model_name, "steps": steps, "observation_count": len(observations)})
        except Exception:
            pass
    return {"observations": observations}


def _normalize_agent_result(raw: Mapping[str, object] | dict[str, object]) -> dict[str, object]:
    observations = raw.get("observations") if isinstance(raw, Mapping) else []
    if not isinstance(observations, list):
        observations = []

    def _latest_graph_node_count() -> int:
        if not observations:
            return 0
        latest = observations[-1]
        result = latest.get("result") if isinstance(latest, Mapping) else None
        if not isinstance(result, Mapping):
            return 0
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
    if isinstance(raw, dict) and "error" in raw:
        normalized["error"] = raw["error"]
    return normalized


def run_ic_iteration(
    task: dict[str, object],
    score_fn: Callable[..., float],
    steps: int = 5,
    parent_trace: object | None = None,
):
    symbol = _require_str(task, "symbol")
    repo = _require_str(task, "repo")
    agent_task: dict[str, object] = {
        "symbol": symbol,
        "repo": repo,
        "symbol_id": symbol,
        "raw_symbol": symbol,
    }

    backend = IterativeContextBackend(repo=repo, score_fn=score_fn)
    tool_specs: list[OpenAITool] = backend.tool_specs
    system_prompt = _build_ic_system_prompt(tool_specs)
    backend_obs = start_span(parent_trace, "iterative_context", metadata={"backend": "iterative_context", "repo": repo, "symbol": symbol})
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
        if backend_obs and hasattr(backend_obs, "end"):
            try:
                backend_obs.end(error=exc)
            except Exception:
                pass
    normalized = _normalize_agent_result(raw)
    nc_val = normalized.get("node_count", 0)
    node_count = float(nc_val) if isinstance(nc_val, (int, float, bool)) else 0.0
    emit_score_for_handle(backend_obs, name="ic.node_count", value=node_count, data_type="NUMERIC")
    if backend_obs and hasattr(backend_obs, "end"):
        try:
            backend_obs.end(metadata={"node_count": normalized.get("node_count", 0)})
        except Exception:
            pass
    return normalized


def run_jc_iteration(
    task: dict[str, object],
    steps: int = 5,
    parent_trace: object | None = None,
):
    symbol = _require_str(task, "symbol")
    repo = _require_str(task, "repo")
    agent_task: dict[str, object] = {
        "symbol": symbol,
        "repo": repo,
        "symbol_id": symbol,
        "raw_symbol": symbol,
    }

    backend = JCodeMunchBackend(repo=repo)
    tool_specs: list[OpenAITool] = backend.tool_specs
    system_prompt = _build_jc_system_prompt(tool_specs)
    backend_obs = start_span(parent_trace, "jcodemunch", metadata={"backend": "jcodemunch", "repo": repo, "symbol": symbol})
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
        if backend_obs and hasattr(backend_obs, "end"):
            try:
                backend_obs.end(error=exc)
            except Exception:
                pass
    normalized = _normalize_agent_result(raw)
    nc_val = normalized.get("node_count", 0)
    node_count = float(nc_val) if isinstance(nc_val, (int, float, bool)) else 0.0
    emit_score_for_handle(backend_obs, name="jc.node_count", value=node_count, data_type="NUMERIC")
    if backend_obs and hasattr(backend_obs, "end"):
        try:
            backend_obs.end(metadata={"node_count": normalized.get("node_count", 0)})
        except Exception:
            pass
    return normalized
