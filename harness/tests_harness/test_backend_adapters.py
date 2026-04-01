from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from mcp.types import TextContent, Tool

from harness import runner
from harness.tools.backends.ic_backend import IterativeContextBackend
from harness.tools.backends.jc_backend import JCodeMunchBackend
from harness.tools.mcp_adapter import (
    mcp_tool_to_openai_tool,
    parse_text_content_payload,
    run_async,
    serialize_tool_result_for_model,
)
from harness.utils.openai_schema import OpenAITool


async def _sample_coroutine(loop_ids: list[int] | None = None) -> str:
    if loop_ids is not None:
        loop_ids.append(id(asyncio.get_running_loop()))
    await asyncio.sleep(0)
    return "ok"


def test_run_async_without_running_loop():
    assert run_async(_sample_coroutine()) == "ok"


def test_run_async_with_running_loop_uses_separate_loop():
    loop_ids: list[int] = []

    async def _inner():
        outer_loop_id = id(asyncio.get_running_loop())
        result = run_async(_sample_coroutine(loop_ids))
        return outer_loop_id, result

    outer_loop_id, result = asyncio.run(_inner())
    assert result == "ok"
    assert loop_ids  # captured inner loop id
    assert loop_ids[0] != outer_loop_id


def test_ic_backend_uses_server_surfaces(monkeypatch, tmp_path):
    dummy_tool = Tool(
        name="resolve",
        description="resolve",
        inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}},
    )
    calls: list[tuple[str, dict[str, Any]]] = []
    runtime_args: list[dict[str, Any]] = []

    async def fake_list_tools():
        return [dummy_tool]

    class FakeRuntime:
        def __init__(self, score_fn=None, repo_root=None):
            runtime_args.append({"score_fn": score_fn, "repo_root": repo_root})

        async def call_tool(self, name: str, arguments: dict[str, Any]):
            calls.append((name, arguments))
            return [TextContent(type="text", text=json.dumps({"ok": True, "name": name}))]

    monkeypatch.setattr("harness.tools.backends.ic_backend.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.tools.backends.ic_backend.IterativeContextToolRuntime", FakeRuntime)

    def score_fn(a, b):
        return 1.0

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    cwd_before = Path.cwd()

    backend = IterativeContextBackend(repo=str(repo_path), score_fn=score_fn)

    assert backend.tool_specs == [mcp_tool_to_openai_tool(dummy_tool)]
    assert runtime_args == [{"score_fn": score_fn, "repo_root": repo_path.resolve()}]
    assert Path.cwd() == cwd_before

    result = backend.dispatch("resolve", {"symbol": "foo"})
    assert result == {"ok": True, "name": "resolve"}
    assert calls == [("resolve", {"symbol": "foo"})]


def test_jc_backend_uses_server_surfaces(monkeypatch, tmp_path):
    dummy_tool = Tool(
        name="search_symbols",
        description="search",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    call_args: list[tuple[str, dict[str, Any]]] = []

    async def fake_list_tools():
        return [dummy_tool]

    async def fake_call_tool(name: str, arguments: dict[str, Any]):
        call_args.append((name, arguments))
        return [TextContent(type="text", text=json.dumps({"result": "ok", "args": arguments}))]  # type: ignore[return-value]

    monkeypatch.setattr("harness.tools.backends.jc_backend.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.tools.backends.jc_backend.call_tool", fake_call_tool)

    backend = JCodeMunchBackend(repo=str(tmp_path))

    assert backend.tool_specs == [mcp_tool_to_openai_tool(dummy_tool)]

    result = backend.dispatch("search_symbols", {"query": "foo"})
    assert result == {"result": "ok", "args": {"query": "foo"}}
    assert ("search_symbols", {"query": "foo"}) in call_args


def test_parse_text_content_payload_handles_json_and_text():
    payload = {"a": 1}
    blocks = [TextContent(type="text", text=json.dumps(payload))]
    assert parse_text_content_payload(blocks) == payload

    plain = [TextContent(type="text", text="not-json")]
    assert parse_text_content_payload(plain) == "not-json"


def test_serialize_tool_result_for_model_json_and_string():
    payload = {"b": 2, "a": 1}
    serialized = serialize_tool_result_for_model(payload)
    assert serialized == json.dumps(payload, sort_keys=True)

    assert serialize_tool_result_for_model("already") == "already"


def test_normalize_agent_result_prefers_full_graph_then_graph():
    raw = {
        "observations": [
            {"tool": "resolve", "result": {"graph": {"nodes": [1]}}},
            {"tool": "expand", "result": {"graph": {"nodes": [4, 5]}, "full_graph": {"nodes": [1, 2, 3]}}},
        ]
    }
    normalized = runner._normalize_agent_result(raw)  # type: ignore[attr-defined]
    assert normalized["node_count"] == 3

    raw_graph_only = {"observations": [{"tool": "expand", "result": {"graph": {"nodes": [1, 2]}}}]}
    normalized_graph_only = runner._normalize_agent_result(raw_graph_only)  # type: ignore[attr-defined]
    assert normalized_graph_only["node_count"] == 2

    raw_empty = {"observations": [{"tool": "expand", "result": {"graph": {}}}]}
    normalized_empty = runner._normalize_agent_result(raw_empty)  # type: ignore[attr-defined]
    assert normalized_empty["node_count"] == 0


def _dummy_client(tool_name: str, args: dict[str, Any]):
    class DummyFunction:
        def __init__(self):
            self.arguments = json.dumps(args)
            self.name = tool_name

    class DummyToolCall:
        def __init__(self):
            self.function = DummyFunction()
            self.id = "call-1"

    class DummyMessage:
        def __init__(self):
            self.tool_calls = [DummyToolCall()]

    class DummyChoice:
        def __init__(self):
            self.message = DummyMessage()

    class DummyResponse:
        def __init__(self):
            self.choices = [DummyChoice()]

    class DummyCompletions:
        def create(self, **kwargs):
            return DummyResponse()

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = DummyChat()

    return DummyClient(), "dummy-model"


def test_run_ic_iteration_preserves_full_graph_and_score_source(monkeypatch, tmp_path):
    dummy_tool = Tool(
        name="resolve",
        description="resolve",
        inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}},
    )

    async def fake_list_tools():
        return [dummy_tool]

    class FakeRuntime:
        def __init__(self, score_fn=None, repo_root=None):
            self.score_fn = score_fn
            self.repo_root = repo_root

        async def call_tool(self, name: str, arguments: dict[str, Any]):
            payload = {
                "graph": {"nodes": [1]},
                "full_graph": {"nodes": [1, 2]},
                "score_source": "injected",
                "node": {"id": "n1"},
            }
            return [TextContent(type="text", text=json.dumps(payload))]

    monkeypatch.setattr("harness.tools.backends.ic_backend.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.tools.backends.ic_backend.IterativeContextToolRuntime", FakeRuntime)
    monkeypatch.setattr(runner, "_make_client", lambda: _dummy_client("resolve", {"symbol": "foo"}))
    class Span:
        id = "span"

        def score(self, name: str, value: float, metadata=None):
            pass

    monkeypatch.setattr(runner, "start_span", lambda *a, **k: Span())

    result = cast(dict[str, Any], runner.run_ic_iteration({"symbol": "s", "repo": str(tmp_path)}, score_fn=lambda n, s: 1.0, steps=1))
    assert result["node_count"] == 2
    first_obs = cast(dict[str, Any], result["observations"][0])
    payload = cast(dict[str, Any], first_obs["result"])
    assert payload["full_graph"] == {"nodes": [1, 2]}
    assert payload["score_source"] == "injected"
    assert payload["node"] == {"id": "n1"}


def test_run_jc_iteration_uses_call_tool(monkeypatch, tmp_path):
    dummy_tool = Tool(
        name="search_symbols",
        description="search",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    async def fake_list_tools():
        return [dummy_tool]

    async def fake_call_tool(name: str, arguments: dict[str, Any]):
        return [TextContent(type="text", text=json.dumps({"called": name, "args": arguments}))]  # type: ignore[return-value]

    monkeypatch.setattr("harness.tools.backends.jc_backend.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.tools.backends.jc_backend.call_tool", fake_call_tool)
    monkeypatch.setattr(runner, "_make_client", lambda: _dummy_client("search_symbols", {"query": "foo"}))
    class Span:
        id = "span"

        def score(self, name: str, value: float, metadata=None):
            pass

    monkeypatch.setattr(runner, "start_span", lambda *a, **k: Span())

    result = cast(dict[str, Any], runner.run_jc_iteration({"symbol": "s", "repo": str(tmp_path)}, steps=1))
    assert result["node_count"] == 0
    first_obs = cast(dict[str, Any], result["observations"][0])
    assert first_obs["result"] == {"called": "search_symbols", "args": {"query": "foo"}}


def test_jc_prompt_uses_canonical_tool_names():
    specs: list[OpenAITool] = [
        {"type": "function", "function": {"name": "alpha", "description": "", "parameters": {}}},
        {"type": "function", "function": {"name": "beta", "description": "", "parameters": {}}},
    ]
    prompt = runner._build_jc_system_prompt(specs)  # type: ignore[attr-defined]
    assert "- alpha" in prompt
    assert "- beta" in prompt


def test_jc_backend_initializes_github_url(monkeypatch, tmp_path):
    dummy_tool = Tool(name="search_symbols", description="search", inputSchema={})
    called: dict[str, int] = {"index_repo": 0}

    async def fake_list_tools():
        return [dummy_tool]

    async def fake_call_tool(name: str, arguments: dict[str, Any]):
        if name == "index_repo":
            called["index_repo"] += 1
            assert arguments["url"] == "https://github.com/owner/repo"
            return [TextContent(type="text", text=json.dumps({"repo": "id"}))]
        return []

    monkeypatch.setattr("harness.tools.backends.jc_backend.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.tools.backends.jc_backend.call_tool", fake_call_tool)

    JCodeMunchBackend(repo="https://github.com/owner/repo")
    assert called["index_repo"] == 1


def test_jc_backend_owner_repo_init(monkeypatch):
    dummy_tool = Tool(name="search_symbols", description="search", inputSchema={})
    called: dict[str, int] = {"index_repo": 0}

    async def fake_list_tools():
        return [dummy_tool]

    async def fake_call_tool(name: str, arguments: dict[str, Any]):
        if name == "index_repo":
            called["index_repo"] += 1
            assert arguments["url"] == "owner/repo"
            return [TextContent(type="text", text=json.dumps({"repo": "id"}))]
        return []

    monkeypatch.setattr("harness.tools.backends.jc_backend.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.tools.backends.jc_backend.call_tool", fake_call_tool)

    JCodeMunchBackend(repo="owner/repo")
    assert called["index_repo"] == 1


def test_jc_backend_rejects_ambiguous_repo(monkeypatch):
    async def fake_list_tools():
        return []

    monkeypatch.setattr("harness.tools.backends.jc_backend.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.tools.backends.jc_backend.call_tool", lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError):
        JCodeMunchBackend(repo="owner/repo/extra")


def test_jc_backend_initializes_local_repo(monkeypatch, tmp_path):
    dummy_tool = Tool(name="search_symbols", description="search", inputSchema={})
    called: dict[str, int] = {"resolve": 0, "index": 0}

    async def fake_list_tools():
        return [dummy_tool]

    async def fake_call_tool(name: str, arguments: dict[str, Any]):
        if name == "resolve_repo":
            called["resolve"] += 1
            return [TextContent(type="text", text=json.dumps({"error": "missing"}))]
        if name == "index_folder":
            called["index"] += 1
            return [TextContent(type="text", text=json.dumps({"repo": "repo-id"}))]
        return []

    monkeypatch.setattr("harness.tools.backends.jc_backend.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.tools.backends.jc_backend.call_tool", fake_call_tool)

    backend = JCodeMunchBackend(repo=str(tmp_path))
    assert called["resolve"] == 1
    assert called["index"] == 1
    assert backend.tool_specs == [mcp_tool_to_openai_tool(dummy_tool)]
