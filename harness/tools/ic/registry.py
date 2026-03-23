from __future__ import annotations

from . import resolve, expand, resolve_and_expand
from ..schema import Tool

IC_TOOLS = [
    Tool(
        name="resolve",
        func=resolve,
        description="Resolve a symbol into a graph node",
        parameters={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    ),
    Tool(
        name="expand",
        func=expand,
        description="Expand a node in the graph",
        parameters={
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "depth": {"type": "integer"},
            },
            "required": ["node_id", "depth"],
        },
    ),
    Tool(
        name="resolve_and_expand",
        func=resolve_and_expand,
        description="Resolve a symbol and expand its graph",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "depth": {"type": "integer"},
            },
            "required": ["symbol", "depth"],
        },
    ),
]
