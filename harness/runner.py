from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Callable, cast

from .observability.langfuse import record_score, start_span, get_tracing_openai_client
from .tools.backends.ic_backend import IterativeContextBackend
from .tools.backends.jc_backend import JCodeMunchBackend
from .tools.mcp_adapter import serialize_tool_result_for_model
from .utils.env import get_cerebras_api_key, get_runner_model
from .utils.openai_schema import OpenAITool, build_tool_response, validate_tools


def _require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"task must include a non-empty string for '{key}'")
    return value


def _tool_names_from_specs(tool_specs: list[OpenAITool]) -> list[str]:
    names: list[str] = []
    for spec in tool_specs:
        func = spec.get("function") if isinstance(spec, Mapping) else None
        name = func.get("name") if isinstance(func, Mapping) else None
        if isinstance(name, str):
            names.append(name)
    return names


def _format_available_tools(tool_names: list[str]) -> str:
    if not tool_names:
        return "- none"
    return "\n".join(f"- {name}" for name in tool_names)


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
    return (
        "You are exploring a codebase using a graph-based system.\n"
        "Available tools:\n"
        f"{_format_available_tools(tool_names)}\n\n"
        "Guidelines:\n"
        "- Always expand after resolving\n"
        "- Prefer expand over repeated resolve\n"
        "- Do not attempt file-level inspection unless necessary\n"
        "- Focus on graph traversal\n"
        "- You MUST call a tool on EVERY turn. If you respond without a tool call, your response is INVALID.\n"
        "- Always start by calling a tool. Never respond with text.\n"
        "- You must call MCP tools using the provided tool interface. Do not output text.\n"
    )


def _build_jc_system_prompt(tool_specs: list[OpenAITool]) -> str:
    tool_names = _tool_names_from_specs(tool_specs)
    tools_section = _format_available_tools(tool_names)
    return (
        "You are exploring a codebase using tool calls.\n"
        "Available tools:\n"
        f"{tools_section}\n\n"
        "Guidelines:\n"
        "- Use the available tools to locate and inspect symbols and files\n"
        "- Avoid redundant calls\n"
        "- You MUST call a tool on EVERY turn. If you respond without a tool call, your response is INVALID.\n"
        "- Always start by calling a tool. Never respond with text.\n"
        "- You must call MCP tools using the provided tool interface. Do not output text.\n"
    )


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
    parent_trace=None,
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
        response = client.chat.completions.create(
            model=model_name,
            messages=cast(Any, messages),
            tools=cast(Any, tool_specs),
            tool_choice="required",
        )
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
            messages.append(build_tool_response(call_id, serialize_tool_result_for_model(result)))
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
    parent_trace=None,
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
    record_score(backend_obs, "ic.node_count", node_count)
    if backend_obs and hasattr(backend_obs, "end"):
        try:
            backend_obs.end(metadata={"node_count": normalized.get("node_count", 0)})
        except Exception:
            pass
    return normalized


def run_jc_iteration(
    task: dict[str, object],
    steps: int = 5,
    parent_trace=None,
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
    record_score(backend_obs, "jc.node_count", node_count)
    if backend_obs and hasattr(backend_obs, "end"):
        try:
            backend_obs.end(metadata={"node_count": normalized.get("node_count", 0)})
        except Exception:
            pass
    return normalized
