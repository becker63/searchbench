from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Iterable
from typing import Any, TypeVar

from mcp.types import TextContent, Tool as MCPTool

from harness.utils.openai_schema import ChatCompletionToolParam, validate_tools

T = TypeVar("T")


def _normalize_parameters(schema: Any) -> dict[str, Any]:
    """
    Coerce MCP inputSchema into an object-shaped JSON schema for OpenAI tools.
    Falls back to a safe empty object when the upstream schema is missing or malformed.
    """
    default: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
    if not isinstance(schema, dict):
        return default
    normalized = dict(schema)
    normalized["type"] = "object"
    properties = normalized.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    normalized["properties"] = properties
    normalized["additionalProperties"] = False
    if "required" not in normalized and properties:
        normalized["required"] = list(properties.keys())
    return normalized


def mcp_tool_to_openai_tool(tool: MCPTool) -> ChatCompletionToolParam:
    """Convert an MCP Tool definition into the OpenAI function tool schema."""
    parameters = _normalize_parameters(tool.inputSchema)
    strict_safe = (
        parameters.get("additionalProperties") is False
        and isinstance(parameters.get("properties"), dict)
        and all(isinstance(k, str) for k in parameters["properties"].keys())
    )
    tool_spec: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": parameters,
        },
    }
    if strict_safe and parameters.get("properties"):
        tool_spec["function"]["strict"] = True
    validate_tools([tool_spec])
    return tool_spec


def parse_text_content_payload(blocks: Iterable[TextContent] | None) -> Any:
    """
    Parse the first text content block as JSON when possible, otherwise return raw text.

    Tool runtimes return a list[TextContent]; harness wants Python payloads in observations.
    """
    if not blocks:
        return None

    for block in blocks:
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            return json.loads(text)
        except Exception:
            return text
    return None


def run_async(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine from sync code safely.

    - If no event loop is running: run with asyncio.run.
    - If a loop is already running on this thread: execute the coroutine inside a
      fresh event loop on a dedicated background thread to avoid deadlocks.
    """
    async def _awaitable() -> T:
        return await coro

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_awaitable())

    result: list[T] = []
    error: list[BaseException] = []

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result.append(loop.run_until_complete(_awaitable()))
        except BaseException as exc:  # pragma: no cover - exercised in tests
            error.append(exc)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if error:
        raise error[0]
    return result[0]


def serialize_tool_result_for_model(result: Any) -> str:
    """
    Serialize tool output for inclusion in the chat transcript.

    Keeps strings as-is; attempts deterministic JSON for structured data.
    """
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, sort_keys=True)
    except Exception:
        return str(result)
