from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Callable

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from iterative_context import exploration
from iterative_context.graph_models import Graph, GraphNode
from iterative_context.server import create_mcp_server
from iterative_context.types import SelectionCallable
from jcodemunch_mcp.server import run_streamable_http_server


def _require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"task must include a non-empty string for '{key}'")
    return value


async def _jc_via_server(repo: str, symbol_id: str, host: str, port: int) -> dict[str, Any]:
    server_task = asyncio.create_task(run_streamable_http_server(host, port))
    try:
        await asyncio.sleep(0.2)  # brief startup delay
        async with streamable_http_client(f"http://{host}:{port}/mcp") as (read_stream, write_stream, _):
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            result: CallToolResult = await session.call_tool(
                "get_context_bundle",
                {"repo": repo, "symbol_id": symbol_id, "output_format": "json"},
            )
            if not result.content:
                return {"error": "empty response"}
            first = result.content[0]
            if isinstance(first, TextContent):
                try:
                    return json.loads(first.text)
                except json.JSONDecodeError:
                    return {"raw": first.text}
            if isinstance(first, dict):
                payload = first.get("text")
                if isinstance(payload, str):
                    try:
                        return json.loads(payload)
                    except json.JSONDecodeError:
                        return {"raw": payload}
            return {"raw": result.model_dump()}
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


def _call_jcodemunch(repo: str, symbol_id: str) -> dict[str, Any]:
    return asyncio.run(_jc_via_server(repo, symbol_id, host="127.0.0.1", port=8901))


def run_both(task: dict[str, object], score_fn: Callable[[object, object], float]):
    """
    Run both iterative-context and jcodemunch MCP servers against the same symbol.

    The injected score_fn is wired into iterative-context via the MCP server
    creation to influence traversal.
    """
    symbol = _require_str(task, "symbol")
    repo = _require_str(task, "repo")
    depth_raw = task.get("depth", 2)
    if isinstance(depth_raw, (int, str)):
        try:
            depth = int(depth_raw)
        except (TypeError, ValueError):
            depth = 2
    else:
        depth = 2

    # Adapt the policy to the traversal signature.
    def ic_score(node: GraphNode, graph: Graph, step: int) -> float:
        state: dict[str, object] = {"graph": graph, "step": step}
        return float(score_fn(node, state))

    traversal_score: SelectionCallable = ic_score

    # Instantiate MCP servers to mirror how agents will use them.
    create_mcp_server(score_fn=traversal_score)
    exploration.ensure_graph_loaded()
    ic_graph = exploration.resolve_and_expand(symbol, depth=depth, score_fn=traversal_score)

    symbol_id = str(task.get("symbol_id") or symbol)
    jc_result = _call_jcodemunch(repo, symbol_id)

    return {
        "iterative_context": {"graph": ic_graph, "symbol": symbol, "depth": depth},
        "jcodemunch": {"bundle": jc_result, "symbol": symbol_id, "repo": repo},
    }
