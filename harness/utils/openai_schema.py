from __future__ import annotations

import json
from typing import Literal, TypedDict


class OpenAIToolFunction(TypedDict):
    name: str
    description: str
    parameters: dict


class OpenAITool(TypedDict):
    type: Literal["function"]
    function: OpenAIToolFunction


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict


def build_tool_response(call_id: str, content: str) -> dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
    }


def parse_arguments(raw: str | dict) -> dict:
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def validate_tools(tools: list[OpenAITool]) -> None:
    assert all("type" in t and "function" in t for t in tools)
