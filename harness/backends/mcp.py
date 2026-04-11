from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Iterable
from typing import Any, TypeVar

from mcp.types import TextContent, Tool as MCPTool

from harness.utils.openai_schema import OpenAITool

T = TypeVar("T")


def mcp_tool_to_openai_tool(tool: MCPTool) -> OpenAITool:
    """Convert an MCP Tool definition into the OpenAI function tool schema."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


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
