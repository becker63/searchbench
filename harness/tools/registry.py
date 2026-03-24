from __future__ import annotations

from typing import Dict, List

from .ic.registry import IC_TOOLS
from .jc.registry import JC_TOOLS
from .schema import Tool, ToolSpec

ALL_TOOLS: List[Tool] = IC_TOOLS + JC_TOOLS
TOOL_MAP: Dict[str, Tool] = {tool.name: tool for tool in ALL_TOOLS}
IC_TOOL_MAP: Dict[str, Tool] = {tool.name: tool for tool in IC_TOOLS}
JC_TOOL_MAP: Dict[str, Tool] = {tool.name: tool for tool in JC_TOOLS}


def get_tool_specs() -> List[ToolSpec]:
    return [tool.to_spec() for tool in ALL_TOOLS]


def get_ic_tool_specs() -> List[ToolSpec]:
    return [tool.to_spec() for tool in IC_TOOLS]


def get_jc_tool_specs() -> List[ToolSpec]:
    return [tool.to_spec() for tool in JC_TOOLS]
