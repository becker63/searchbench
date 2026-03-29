from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Callable, cast

from openai import OpenAI

from .tools.backends.ic_backend import IterativeContextBackend
from .tools.backends.jc_backend import JCodeMunchBackend
from .tools.mcp_adapter import serialize_tool_result_for_model
from .utils.env import load_env
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


def _make_client() -> OpenAI:
    load_env()
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing CEREBRAS_API_KEY in environment")
    model = os.environ.get("RUNNER_MODEL", "llama3.1-8b")
    client = OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
    )
    print(f"[LLM] Runner model: {model}")
    return client


def run_agent(
    task: dict[str, object],
    steps: int,
    tool_specs: list[OpenAITool],
    dispatch_tool_call: Callable[[str, dict[str, Any]], Any],
    system_prompt: str,
) -> dict[str, object]:
    """
    Execute a deterministic tool-only agent for a fixed number of steps.
    """
    client = _make_client()
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
    model = os.environ.get("RUNNER_MODEL", "llama3.1-8b")
    for _ in range(steps):
        response = client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            tools=cast(Any, tool_specs),
            tool_choice="required",
        )

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            raise RuntimeError("Model failed to produce tool call")

        call = tool_calls[0]
        call_id = getattr(call, "id", None)
        if not call_id:
            raise RuntimeError("Tool call missing id")
        try:
            raw_args = call.function.arguments or {}
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            tool_name = call.function.name
            result = dispatch_tool_call(tool_name, parsed_args)
            observations.append({"tool": tool_name, "result": result})
            messages.append(build_tool_response(call_id, serialize_tool_result_for_model(result)))
        except Exception as exc:  # noqa: BLE001
            tool_name = call.function.name or "unknown"
            observations.append({"tool": tool_name, "error": str(exc)})
            messages.append(build_tool_response(call_id, f"ERROR: {exc}"))

    if not observations:
        raise RuntimeError("Agent failed to call any tools")

    return {"observations": observations}


def _normalize_agent_result(raw: dict[str, object]) -> dict[str, object]:
    observations = raw.get("observations") if isinstance(raw, dict) else []
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


def run_ic_iteration(task: dict[str, object], score_fn: Callable[..., float], steps: int = 5):
    symbol = _require_str(task, "symbol")
    repo = _require_str(task, "repo")
    agent_task: dict[str, object] = {
        "symbol": symbol,
        "repo": repo,
        "symbol_id": symbol,
        "raw_symbol": symbol,
    }

    backend = IterativeContextBackend(score_fn)
    tool_specs: list[OpenAITool] = backend.tool_specs
    system_prompt = _build_ic_system_prompt(tool_specs)
    try:
        raw = run_agent(
            agent_task,
            steps=steps,
            tool_specs=tool_specs,
            dispatch_tool_call=backend.dispatch,
            system_prompt=system_prompt,
        )
    except Exception as exc:  # noqa: BLE001
        raw = {"observations": [], "error": str(exc)}
    return _normalize_agent_result(raw)


def run_jc_iteration(task: dict[str, object], steps: int = 5):
    symbol = _require_str(task, "symbol")
    repo = _require_str(task, "repo")
    agent_task: dict[str, object] = {
        "symbol": symbol,
        "repo": repo,
        "symbol_id": symbol,
        "raw_symbol": symbol,
    }

    backend = JCodeMunchBackend()
    tool_specs: list[OpenAITool] = backend.tool_specs
    system_prompt = _build_jc_system_prompt(tool_specs)
    try:
        raw = run_agent(
            agent_task,
            steps=steps,
            tool_specs=tool_specs,
            dispatch_tool_call=backend.dispatch,
            system_prompt=system_prompt,
        )
    except Exception as exc:  # noqa: BLE001
        raw = {"observations": [], "error": str(exc)}
    return _normalize_agent_result(raw)
