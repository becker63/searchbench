from __future__ import annotations

import json
from typing import Any, Dict

from .registry import TOOL_MAP


def parse_tool_call(text: str) -> Dict[str, Any]:
    """
    Expected JSON format:
    {"tool": "<name>", "arguments": {...}}
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    tool_name = data.get("tool")
    args = data.get("arguments", {})

    if tool_name not in TOOL_MAP:
        raise ValueError(f"Unknown tool: {tool_name}")

    return {"tool": tool_name, "arguments": args}


def execute_tool_call(call: Dict[str, Any], score_fn: Any | None = None):
    tool = TOOL_MAP[call["tool"]]
    func = tool.func

    if "score_fn" in func.__code__.co_varnames:
        return func(**call["arguments"], score_fn=score_fn)

    return func(**call["arguments"])
