from __future__ import annotations

import json
import os
from typing import Any, Callable, cast

from openai import OpenAI

from .utils.openai_schema import OpenAITool, build_tool_response, validate_tools
from .tools.registry import (
    get_ic_tool_specs,
    get_jc_tool_specs,
    IC_TOOL_MAP,
    JC_TOOL_MAP,
)
from .utils.env import load_env


def _require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"task must include a non-empty string for '{key}'")
    return value


def _dispatch_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    tool_map: dict[str, Any],
    score_fn: Any,
) -> Any:
    if tool_name not in tool_map:
        raise ValueError(f"Unknown tool: {tool_name}")
    tool = tool_map[tool_name]
    func = tool.func

    args = dict(arguments)
    if "depth" in args:
        try:
            args["depth"] = int(args["depth"])
        except Exception:
            args["depth"] = 1
    if score_fn is not None:
        try:
            return func(**args, score_fn=score_fn)
        except TypeError:
            pass
    return func(**args)


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
    score_fn: Any,
    steps: int,
    tool_specs: list[OpenAITool],
    tool_map: dict[str, Any],
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
            result = _dispatch_tool_call(
                tool_name=tool_name,
                arguments=parsed_args,
                tool_map=tool_map,
                score_fn=score_fn,
            )
            observations.append({"tool": tool_name, "result": result})
            messages.append(build_tool_response(call_id, str(result)))
        except Exception as exc:  # noqa: BLE001
            tool_name = call.function.name or "unknown"
            observations.append({"tool": tool_name, "error": str(exc)})
            messages.append(build_tool_response(call_id, f"ERROR: {exc}"))

    if not observations:
        raise RuntimeError("Agent failed to call any tools")

    return {"observations": observations}


def run_ic_iteration(task: dict[str, object], score_fn: Callable[[object, object], float], steps: int = 5):
    symbol = _require_str(task, "symbol")
    repo = _require_str(task, "repo")
    agent_task: dict[str, object] = {
        "symbol": symbol,
        "repo": repo,
        "symbol_id": symbol,
        "raw_symbol": symbol,
    }

    tool_specs: list[OpenAITool] = get_ic_tool_specs()
    system_prompt = (
        "You are exploring a codebase using a graph-based system.\n"
        "Available tools:\n"
        "- resolve\n- expand\n- resolve_and_expand\n\n"
        "Guidelines:\n"
        "- Always expand after resolving\n"
        "- Prefer expand over repeated resolve\n"
        "- Do not attempt file-level inspection unless necessary\n"
        "- Focus on graph traversal\n"
        "- You MUST call a tool on EVERY turn. If you respond without a tool call, your response is INVALID.\n"
        "- Always start by calling a tool. Never respond with text.\n"
        "- You must call MCP tools using the provided tool interface. Do not output text.\n"
    )
    try:
        raw = run_agent(
            agent_task,
            score_fn=score_fn,
            steps=steps,
            tool_specs=tool_specs,
            tool_map=IC_TOOL_MAP,
            system_prompt=system_prompt,
        )
    except Exception as exc:  # noqa: BLE001
        raw = {"observations": [], "error": str(exc)}
    return normalize_iterative_context(raw)


def run_jc_iteration(task: dict[str, object], steps: int = 5):
    symbol = _require_str(task, "symbol")
    repo = _require_str(task, "repo")
    agent_task: dict[str, object] = {
        "symbol": symbol,
        "repo": repo,
        "symbol_id": symbol,
        "raw_symbol": symbol,
    }

    tool_specs: list[OpenAITool] = get_jc_tool_specs()
    system_prompt = (
        "You are exploring a codebase using tool calls.\n"
        "Available tools:\n"
        "- search_symbols\n- get_symbol\n- get_file_outline\n- find_references\n- get_file_tree\n\n"
        "Guidelines:\n"
        "- Use search to locate symbols\n"
        "- Inspect files to understand structure\n"
        "- Use references/imports to navigate\n"
        "- Avoid redundant calls\n"
        "- You MUST call a tool on EVERY turn. If you respond without a tool call, your response is INVALID.\n"
        "- Always start by calling a tool. Never respond with text.\n"
        "- You must call MCP tools using the provided tool interface. Do not output text.\n"
    )
    return run_agent(
        agent_task,
        score_fn=None,
        steps=steps,
        tool_specs=tool_specs,
        tool_map=JC_TOOL_MAP,
        system_prompt=system_prompt,
    )
