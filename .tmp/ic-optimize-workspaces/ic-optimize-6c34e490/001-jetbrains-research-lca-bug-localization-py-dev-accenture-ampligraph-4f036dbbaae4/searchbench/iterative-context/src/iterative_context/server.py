from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

from mcp.server import Server
from mcp.types import TextContent, Tool

from iterative_context import exploration
from iterative_context.scoring import default_score_fn
from iterative_context.serialization import serialize_graph, serialize_node
from iterative_context.types import SelectionCallable

server = Server("iterative-context")


def _resolve_symbol(symbol: str) -> dict[str, Any] | None:
    """Look up a symbol and serialize it with any graph extras."""
    node = exploration.resolve(symbol)
    if node is None:
        return None

    graph = exploration.get_active_graph()
    extras: dict[str, Any] = {}
    if graph is not None and node.id in graph.nodes:
        raw = graph.nodes[node.id]
        extras = {k: v for k, v in raw.items() if k != "data"}

    return serialize_node(node, extras)


def _serialize_active_graph() -> dict[str, Any]:
    graph = exploration.get_active_graph()
    metadata: dict[str, Any] = {"source": "active_graph"}
    metadata.update(exploration.get_active_repo_metadata())
    if graph is None:
        return {"nodes": [], "edges": [], "metadata": metadata}
    return serialize_graph(graph, metadata=metadata)


def _tool_definitions() -> list[Tool]:
    return [
        Tool(
            name="resolve",
            description="Resolve a symbol to a graph node if present.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol identifier to resolve"}
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="expand",
            description="Expand a node outward to a bounded depth.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the node to expand"},
                    "depth": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Expansion depth to explore",
                    },
                },
                "required": ["node_id", "depth"],
            },
        ),
        Tool(
            name="resolve_and_expand",
            description="Resolve a symbol and expand its neighborhood.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol identifier to resolve"},
                    "depth": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Expansion depth to explore",
                    },
                },
                "required": ["symbol", "depth"],
            },
        ),
    ]


def _as_text_content(payload: dict[str, Any]) -> TextContent:
    return TextContent(type="text", text=json.dumps(payload))


class IterativeContextToolRuntime:
    """In-process MCP-compatible runtime with optional scoring injection."""

    def __init__(
        self,
        score_fn: SelectionCallable | None = None,
        repo_root: str | Path | None = None,
    ):
        self._score_fn = score_fn
        self._repo_root = Path(repo_root).resolve() if repo_root is not None else None

    async def list_tools(self) -> list[Tool]:
        return _tool_definitions()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            result = self._dispatch(name, arguments or {})
            return [_as_text_content(result)]
        except Exception as exc:  # pragma: no cover - defensive guard
            return [_as_text_content({"error": str(exc)})]

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        score_fn, score_source = _resolve_effective_score_fn(self._score_fn)
        tool_name = name.strip()
        if tool_name == "resolve":
            symbol = _require_str(arguments, "symbol")
            self._ensure_graph_ready()
            return {
                "node": _resolve_symbol(symbol),
                "full_graph": _serialize_active_graph(),
                "score_source": score_source,
            }

        if tool_name == "expand":
            node_id = _require_str(arguments, "node_id")
            depth = _require_int(arguments, "depth")
            self._ensure_graph_ready()
            graph = exploration.expand(node_id=node_id, depth=depth, score_fn=score_fn)
            return {
                "graph": graph,
                "full_graph": _serialize_active_graph(),
                "score_source": score_source,
            }

        if tool_name == "resolve_and_expand":
            symbol = _require_str(arguments, "symbol")
            depth = _require_int(arguments, "depth")
            self._ensure_graph_ready()

            node = _resolve_symbol(symbol)
            if node is None:
                return {
                    "node": None,
                    "graph": {"nodes": [], "edges": [], "metadata": {}},
                    "full_graph": _serialize_active_graph(),
                    "score_source": score_source,
                }

            return {
                "node": node,
                "graph": exploration.expand(node["id"], depth=depth, score_fn=score_fn),
                "full_graph": _serialize_active_graph(),
                "score_source": score_source,
            }

        return {"error": f"Unknown tool: {name}", "score_source": score_source}

    def _ensure_graph_ready(self) -> None:
        exploration.ensure_graph_loaded(repo_root=self._repo_root)


def _require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_int(arguments: dict[str, Any], key: str) -> int:
    value = arguments.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def load_local_policy() -> SelectionCallable | None:
    """Dynamically load a local policy module if present."""
    try:
        policy_module = importlib.import_module("iterative_context.policy")
    except Exception:
        return None

    candidate = getattr(policy_module, "score_fn", None) or getattr(policy_module, "score", None)
    if callable(candidate):
        return cast(SelectionCallable, candidate)
    return None


def _resolve_effective_score_fn(
    injected_score_fn: SelectionCallable | None,
) -> tuple[SelectionCallable, str]:
    """Resolve the scoring function with explicit precedence."""
    if injected_score_fn is not None:
        return injected_score_fn, "injected"

    local_policy = load_local_policy()
    if local_policy is not None:
        return local_policy, "local_policy"

    return default_score_fn, "default"


_default_runtime = IterativeContextToolRuntime()


async def list_tools() -> list[Tool]:
    return await _default_runtime.list_tools()


async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    return await _default_runtime.call_tool(name, arguments)


@server.list_tools()
async def _server_list_tools() -> list[Tool]:  # pyright: ignore[reportUnusedFunction]
    return await list_tools()


@server.call_tool()
async def _server_call_tool(  # pyright: ignore[reportUnusedFunction]
    name: str, arguments: dict[str, Any]
) -> list[TextContent]:
    return await call_tool(name, arguments)


__all__ = [
    "server",
    "IterativeContextToolRuntime",
    "list_tools",
    "call_tool",
]
