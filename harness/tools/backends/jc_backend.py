from __future__ import annotations

from typing import Any

from jcodemunch_mcp.server import call_tool, list_tools

from ..mcp_adapter import mcp_tool_to_openai_tool, parse_text_content_payload, run_async
from ...utils.openai_schema import OpenAITool


class JCodeMunchBackend:
    def __init__(self) -> None:
        tools = run_async(list_tools())
        self.tool_specs: list[OpenAITool] = [mcp_tool_to_openai_tool(t) for t in tools]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        result_blocks = run_async(call_tool(name, arguments))
        return parse_text_content_payload(result_blocks)
