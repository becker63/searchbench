from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Any, cast

import pytest
from mcp.types import TextContent, Tool

from harness.agents import localizer as runner
from harness.backends.ic import IterativeContextBackend
from harness.backends.jc import JCodeMunchBackend
from harness.backends.mcp import (
    mcp_tool_to_openai_tool,
    parse_text_content_payload,
    run_async,
    serialize_tool_result_for_model,
)
from harness.localization.models import (
    LCAContext,
    LCAGold,
    LCATask,
    LCATaskIdentity,
    LocalizationPrediction,
    LocalizationRunResult,
)
from harness.prompts import build_jc_system_prompt
from harness.utils.openai_schema import ChatCompletionToolParam, validate_tools


def _runner_task(repo: str) -> LCATask:
    return LCATask(
        identity=LCATaskIdentity(
            dataset_name="lca",
            dataset_config="py",
            dataset_split="dev",
            repo_owner="o",
            repo_name="r",
            base_sha="abc",
        ),
        context=LCAContext(issue_title="bug", issue_body="details"),
        gold=LCAGold(changed_files=["a.py"]),
        repo=repo,
    )


def _runner_result(
    task: LCATask,
    predicted_files: list[str] | None = None,
    reasoning: str | None = None,
    observations: list[dict[str, object]] | None = None,
    source: str | None = None,
) -> LocalizationRunResult:
    return LocalizationRunResult(
        task=task,
        prediction=LocalizationPrediction(
            predicted_files=predicted_files or [],
            reasoning=reasoning,
        ),
        observations=observations or [],
        source=source,
    )


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


def test_mcp_tool_conversion_preserves_object_schema():
    tool = Tool(
        name="resolve_path",
        description="Resolve paths",
        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    spec = mcp_tool_to_openai_tool(tool)
    validate_tools([spec])
    params = spec["function"]["parameters"]
    assert params["type"] == "object"
    assert "path" in params["properties"]
    assert params["additionalProperties"] is False
    assert params.get("required") in (None, [])
    assert spec.get("strict") in (None, False)
    assert spec["function"].get("strict") is False


def test_mcp_tool_conversion_keeps_mcp_tools_non_strict_even_when_required_matches():
    tool = Tool(
        name="resolve_path",
        description="Resolve paths",
        inputSchema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    spec = mcp_tool_to_openai_tool(tool)
    validate_tools([spec])
    params = spec["function"]["parameters"]
    assert params["required"] == ["path"]
    assert spec["function"].get("strict") is False


def test_mcp_tool_conversion_defaults_missing_schema():
    tool = Tool.model_construct(name="noop", description=None, inputSchema=None)
    spec = mcp_tool_to_openai_tool(tool)
    validate_tools([spec])
    params = spec["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"] == {}
    assert params["additionalProperties"] is False
    assert spec.get("strict") in (None, False)
    assert spec["function"].get("strict") is False


def test_mcp_tool_conversion_normalizes_non_object_root():
    tool = Tool(name="arrayish", description="bad schema", inputSchema={"type": "array", "items": {}})
    spec = mcp_tool_to_openai_tool(tool)
    validate_tools([spec])
    params = spec["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"] == {}
    assert params["additionalProperties"] is False
    assert spec.get("strict") in (None, False)
    assert spec["function"].get("strict") is False


def test_mcp_tool_conversion_handles_non_mapping_schema():
    tool = Tool.model_construct(name="nondict", description="bad", inputSchema="oops")  # type: ignore[arg-type]
    spec = mcp_tool_to_openai_tool(tool)
    validate_tools([spec])
    params = spec["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"] == {}
    assert params["additionalProperties"] is False
    assert spec.get("strict") in (None, False)
    assert spec["function"].get("strict") is False


def test_mcp_tool_conversion_forces_additional_properties_false():
    tool = Tool(
        name="with_extra_props",
        description="schema has additionalProperties true upstream",
        inputSchema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "repo": {"type": "string"}},
            "additionalProperties": True,
        },
    )
    spec = mcp_tool_to_openai_tool(tool)
    validate_tools([spec])
    params = spec["function"]["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    assert params.get("required") in (None, [])
    assert spec.get("strict") in (None, False)
    assert spec["function"].get("strict") is False


def test_mcp_tool_conversion_preserves_index_repo_optional_fields_without_strict():
    tool = Tool(
        name="index_repo",
        description="Index a GitHub repository",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "use_ai_summaries": {"type": "boolean", "default": True},
                "extra_ignore_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "incremental": {"type": "boolean", "default": True},
            },
            "required": ["url"],
        },
    )
    spec = mcp_tool_to_openai_tool(tool)
    validate_tools([spec])
    params = spec["function"]["parameters"]
    assert params["required"] == ["url"]
    assert set(params["properties"]) == {
        "url",
        "use_ai_summaries",
        "extra_ignore_patterns",
        "incremental",
    }
    assert params["additionalProperties"] is False
    assert spec["function"].get("strict") is False


def test_validate_tools_accepts_function_strict():
    spec: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "do_thing",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"depth": {"type": "integer"}},
                "required": ["depth"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    validate_tools([spec])
    assert spec.get("strict") in (None, False)
    assert spec["function"].get("strict") is True


def test_ic_backend_uses_server_surfaces(monkeypatch, tmp_path):
    dummy_tool = Tool(
        name="resolve_path",
        description="resolve",
        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
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

    monkeypatch.setattr("harness.backends.ic.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.backends.ic.IterativeContextToolRuntime", FakeRuntime)

    def score_fn(a, b):
        return 1.0

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    cwd_before = Path.cwd()

    backend = IterativeContextBackend(repo=str(repo_path), score_fn=score_fn)

    assert backend.tool_specs == [mcp_tool_to_openai_tool(dummy_tool)]
    assert runtime_args == [{"score_fn": score_fn, "repo_root": repo_path.resolve()}]
    assert Path.cwd() == cwd_before

    result = backend.dispatch("resolve_path", {"path": "foo"})
    assert result == {"ok": True, "name": "resolve_path"}
    assert calls == [("resolve_path", {"path": "foo"})]


def test_ic_backend_coerces_integer_arguments(monkeypatch, tmp_path):
    dummy_tool = Tool(
        name="expand",
        description="expand node",
        inputSchema={
            "type": "object",
            "properties": {"node_id": {"type": "string"}, "depth": {"type": "integer"}},
        },
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_list_tools():
        return [dummy_tool]

    class FakeRuntime:
        def __init__(self, score_fn=None, repo_root=None):
            pass

        async def call_tool(self, name: str, arguments: dict[str, Any]):
            calls.append((name, arguments))
            return [TextContent(type="text", text=json.dumps({"ok": True, "args": arguments}))]  # type: ignore[return-value]

    monkeypatch.setattr("harness.backends.ic.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.backends.ic.IterativeContextToolRuntime", FakeRuntime)

    backend = IterativeContextBackend(repo=str(tmp_path), score_fn=None)
    result = backend.dispatch("expand", {"node_id": "n1", "depth": "2"})
    assert result["ok"] is True
    assert calls == [("expand", {"node_id": "n1", "depth": 2})]


def test_jc_backend_uses_server_surfaces(monkeypatch, tmp_path):
    dummy_tool = Tool(
        name="search_files",
        description="search",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    call_args: list[tuple[str, dict[str, Any]]] = []

    async def fake_list_tools():
        return [dummy_tool]

    async def fake_call_tool(name: str, arguments: dict[str, Any]):
        call_args.append((name, arguments))
        return [TextContent(type="text", text=json.dumps({"result": "ok", "args": arguments}))]  # type: ignore[return-value]

    monkeypatch.setattr("harness.backends.jc.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.backends.jc.call_tool", fake_call_tool)

    backend = JCodeMunchBackend(repo=str(tmp_path))

    assert backend.tool_specs == [mcp_tool_to_openai_tool(dummy_tool)]

    result = backend.dispatch("search_files", {"query": "foo"})
    assert result == {"result": "ok", "args": {"query": "foo"}}
    assert ("search_files", {"query": "foo"}) in call_args


def test_jc_backend_initializes_with_index_repo_optional_schema(monkeypatch, tmp_path):
    index_repo_tool = Tool(
        name="index_repo",
        description="index",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "use_ai_summaries": {"type": "boolean", "default": True},
                "extra_ignore_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "incremental": {"type": "boolean", "default": True},
            },
            "required": ["url"],
        },
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_list_tools():
        return [index_repo_tool]

    async def fake_call_tool(name: str, arguments: dict[str, Any]):
        calls.append((name, arguments))
        return [TextContent(type="text", text=json.dumps({"result": "ok"}))]  # type: ignore[return-value]

    monkeypatch.setattr("harness.backends.jc.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.backends.jc.call_tool", fake_call_tool)

    backend = JCodeMunchBackend(repo=str(tmp_path))

    assert backend.tool_specs[0]["function"]["name"] == "index_repo"
    assert backend.tool_specs[0]["function"].get("strict") is False
    assert ("resolve_repo", {"path": str(tmp_path)}) in calls


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


def test_clean_observations_for_finalization_filters_errors_and_paths():
    repo_root = "/tmp/repo"
    observations = [
        {"result": {"error": "depth must be an integer"}},
        {"content": {"error": "depth must be an integer"}, "result": {"error": "depth must be an integer"}},
        {"result": {"file": f"{repo_root}/src/main.py", "notes": "ok"}},
        {"result": {"file": f"{repo_root}/src/main.py", "notes": "dup"}},
        {"result": {"path": "/other/place/file.txt"}},
    ]
    cleaned = runner._clean_observations_for_finalization(observations, repo_root)  # type: ignore[attr-defined]
    assert cleaned
    # Error-only observations are removed.
    assert all("error" not in (obs.get("result") or {}) for obs in cleaned)
    # Paths are normalized and deduplicated.
    files = [obs["result"]["file"] for obs in cleaned if "result" in obs and "file" in obs["result"]]
    assert files == ["src/main.py"]
    paths = [obs["result"]["path"] for obs in cleaned if "result" in obs and "path" in obs["result"]]
    assert paths == ["/other/place/file.txt"]


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
            self.usage = type(
                "U",
                (),
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {},
                    "completion_tokens_details": {},
                },
            )()

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


def test_run_agent_returns_usage_details_for_scoring(monkeypatch):
    updates: list[dict[str, object]] = []

    class DummyMessage:
        content = '{"predicted_files": ["src/app.py"]}'

    class DummyChoice:
        message = DummyMessage()

    class DummyResponse:
        choices = [DummyChoice()]
        usage = type(
            "U",
            (),
            {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
                "prompt_tokens_details": {},
                "completion_tokens_details": {},
            },
        )()
        model = "dummy-model"

    class DummyCompletions:
        def create(self, **kwargs):
            return DummyResponse()

    class DummyClient:
        chat = type("Chat", (), {"completions": DummyCompletions()})()

    @contextmanager
    def span_cm():
        yield type(
            "S",
            (),
            {
                "id": "span",
                "update": lambda self, **kwargs: updates.append(kwargs),
                "end": lambda self, **kwargs: None,
            },
        )()

    monkeypatch.setattr(runner, "_make_client", lambda model_override=None: (DummyClient(), "dummy-model"))
    monkeypatch.setattr(runner, "get_cerebras_api_key", lambda: "key")
    monkeypatch.setattr(runner, "start_observation", lambda *args, **kwargs: span_cm())

    result = runner.run_agent(
        {"identity": {}, "repo": "repo", "context": {}},
        steps=1,
        tool_specs=[],
        dispatch_tool_call=lambda name, args: {},
        system_prompt="system",
        parent_trace=object(),
        backend_name="iterative_context",
    )

    assert result["usage_details"]["total_tokens"] == 16.0
    assert result["step_count"] == 1
    assert result["generation_count"] == 1
    observations = cast(list[dict[str, object]], result["observations"])
    assert observations[0]["usage_details"]["total_tokens"] == 16.0
    assert any("usage_details" in update for update in updates)


def test_run_agent_counts_retry_generation_attempts(monkeypatch):
    class DummyMessage:
        content = '{"predicted_files": ["src/app.py"]}'

    class DummyChoice:
        message = DummyMessage()

    class DummyResponse:
        choices = [DummyChoice()]
        usage = type(
            "U",
            (),
            {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
                "prompt_tokens_details": {},
                "completion_tokens_details": {},
            },
        )()
        model = "dummy-model"

    class DummyCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("context length too long")
            return DummyResponse()

    completions = DummyCompletions()

    class DummyClient:
        chat = type("Chat", (), {"completions": completions})()

    @contextmanager
    def span_cm():
        yield type(
            "S",
            (),
            {
                "id": "span",
                "update": lambda self, **kwargs: None,
                "end": lambda self, **kwargs: None,
            },
        )()

    monkeypatch.setattr(runner, "_make_client", lambda model_override=None: (DummyClient(), "dummy-model"))
    monkeypatch.setattr(runner, "get_cerebras_api_key", lambda: "key")
    monkeypatch.setattr(runner, "start_observation", lambda *args, **kwargs: span_cm())

    result = runner.run_agent(
        {"identity": {}, "repo": "repo", "context": {}},
        steps=1,
        tool_specs=[],
        dispatch_tool_call=lambda name, args: {},
        system_prompt="system",
        parent_trace=object(),
        backend_name="iterative_context",
    )

    assert result["step_count"] == 1
    assert result["generation_count"] == 2


def test_failed_tool_dispatch_counts_as_attempted_failed_call(monkeypatch):
    @contextmanager
    def span_cm():
        yield type(
            "S",
            (),
            {
                "id": "span",
                "update": lambda self, **kwargs: None,
                "end": lambda self, **kwargs: None,
            },
        )()

    monkeypatch.setattr(runner, "_make_client", lambda model_override=None: _dummy_client("search_files", {"query": "foo"}))
    monkeypatch.setattr(runner, "get_cerebras_api_key", lambda: "key")
    monkeypatch.setattr(runner, "start_observation", lambda *args, **kwargs: span_cm())

    def failing_dispatch(name: str, args: dict[str, Any]) -> object:
        raise RuntimeError("tool boom")

    result = runner.run_agent(
        {"identity": {}, "repo": "repo", "context": {}},
        steps=1,
        tool_specs=[
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        dispatch_tool_call=failing_dispatch,
        system_prompt="system",
        parent_trace=object(),
        backend_name="jcodemunch",
    )

    observations = cast(list[dict[str, object]], result["observations"])
    assert runner._tool_stats(observations) == (1, 0, 1)  # type: ignore[attr-defined]


def test_run_ic_iteration_preserves_full_graph_and_score_source(monkeypatch, tmp_path):
    dummy_tool = Tool(
        name="resolve_path",
        description="resolve",
        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
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

    monkeypatch.setattr("harness.backends.ic.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.backends.ic.IterativeContextToolRuntime", FakeRuntime)
    monkeypatch.setattr(runner, "_make_client", lambda: _dummy_client("resolve_path", {"identity": "task"}))
    @contextmanager
    def span_cm():
        class Span:
            id = "span"

            def score(self, name: str, value: float, metadata=None):
                pass

            def update(self, **kwargs):
                pass

            def end(self, **kwargs):
                pass

        yield Span()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())
    def fake_finalize(task, obs, parent_trace=None):
        # Expect a separate no-tools finalize call; simulate schema result.
        assert isinstance(task, LCATask)
        assert isinstance(obs, list)
        # Ensure the schema path is used (response_format=json_schema would have been required upstream).
        return _runner_result(task, ["a.py"], observations=obs, source="finalize")

    monkeypatch.setattr(runner, "_finalize_localization", fake_finalize)

    result = runner.run_ic_iteration(
        _runner_task(str(tmp_path)),
        score_fn=lambda n, s: 1.0,
        steps=1,
        parent_trace=object(),
    )
    assert result.observability_summary is not None
    assert result.observability_summary.backend == "ic"
    assert result.observability_summary.status == "success"
    assert result.observability_summary.step_count == 1
    assert result.observability_summary.generation_count >= 1
    assert result.observability_summary.tool_calls_count == 1
    assert result.observability_summary.successful_tool_calls_count == 1
    assert result.observability_summary.failed_tool_calls_count == 0
    assert result.node_count == 2
    tool_obs = next(
        cast(dict[str, Any], obs)
        for obs in result.observations
        if isinstance(obs, dict) and obs.get("role") == "tool"
    )
    payload = cast(dict[str, Any], tool_obs["result"])
    assert payload["full_graph"] == {"nodes": [1, 2]}
    assert payload["score_source"] == "injected"
    assert payload["node"] == {"id": "n1"}


def test_run_jc_iteration_uses_call_tool(monkeypatch, tmp_path):
    dummy_tool = Tool(
        name="search_files",
        description="search",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    async def fake_list_tools():
        return [dummy_tool]

    async def fake_call_tool(name: str, arguments: dict[str, Any]):
        return [TextContent(type="text", text=json.dumps({"called": name, "args": arguments}))]  # type: ignore[return-value]

    monkeypatch.setattr("harness.backends.jc.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.backends.jc.call_tool", fake_call_tool)
    monkeypatch.setattr(runner, "_make_client", lambda: _dummy_client("search_files", {"query": "foo"}))
    @contextmanager
    def span_cm():
        class Span:
            id = "span"

            def score(self, name: str, value: float, metadata=None):
                pass

            def update(self, **kwargs):
                pass

            def end(self, **kwargs):
                pass

        yield Span()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())
    def fake_finalize(task, obs, parent_trace=None):
        assert isinstance(task, LCATask)
        assert isinstance(obs, list)
        return _runner_result(task, ["a.py"], observations=obs, source="finalize")

    monkeypatch.setattr(runner, "_finalize_localization", fake_finalize)

    result = runner.run_jc_iteration(
        _runner_task(str(tmp_path)),
        steps=1,
        parent_trace=object(),
    )
    assert result.observability_summary is not None
    assert result.observability_summary.backend == "jc"
    assert result.observability_summary.status == "success"
    assert result.observability_summary.step_count == 1
    assert result.observability_summary.generation_count >= 1
    assert result.observability_summary.tool_calls_count == 1
    assert result.observability_summary.successful_tool_calls_count == 1
    assert result.observability_summary.failed_tool_calls_count == 0
    assert result.node_count == 0
    tool_obs = next(
        cast(dict[str, Any], obs)
        for obs in result.observations
        if isinstance(obs, dict) and obs.get("role") == "tool"
    )
    assert tool_obs["result"] == {"called": "search_files", "args": {"query": "foo"}}


def test_run_jc_iteration_traces_backend_init_failure(monkeypatch, tmp_path, caplog):
    updates: list[dict[str, Any]] = []
    starts: list[dict[str, Any]] = []

    async def fake_list_tools():
        raise RuntimeError("schema boom")

    monkeypatch.setattr("harness.backends.jc.list_tools", fake_list_tools)

    @contextmanager
    def span_cm(*args, **kwargs):
        starts.append(dict(kwargs))

        class Span:
            id = "span"

            def score(self, name: str, value: float, metadata=None):
                pass

            def update(self, **kwargs):
                updates.append(dict(kwargs))

            def end(self, **kwargs):
                pass

        yield Span()

    monkeypatch.setattr(runner, "start_observation", span_cm)
    caplog.set_level(logging.INFO, logger="harness.agents.localizer")

    task = _runner_task(str(tmp_path))
    with pytest.raises(Exception, match="schema boom"):
        runner.run_jc_iteration(task, steps=1, parent_trace=object())

    assert starts[0]["name"] == "jcodemunch"
    assert starts[0]["metadata"]["phase"] == "backend_init"
    assert starts[0]["metadata"]["repo"] == str(tmp_path)
    assert starts[0]["metadata"]["identity"] == task.identity.task_id()
    assert updates
    failure_metadata = updates[-1]["metadata"]
    assert failure_metadata["phase"] == "backend_init"
    assert failure_metadata["error"] == "schema boom"
    assert failure_metadata["repo"] == str(tmp_path)
    assert failure_metadata["task_summary"]["stage"] == "backend_init"
    messages = [record.getMessage() for record in caplog.records]
    assert any("[RUN] initializing jc backend" in message for message in messages)
    assert any("jc backend initialization failed" in message for message in messages)


def test_run_ic_iteration_uses_backend_tool_specs(monkeypatch, tmp_path):
    from contextlib import contextmanager

    captured: dict[str, object] = {}
    finalized: dict[str, bool] = {"called": False}
    finalize_calls: list[dict[str, object]] = []

    class FakeBackend:
        def __init__(self, repo: str, score_fn=None):
            captured["repo"] = repo
            self.tool_specs = [
                {
                    "type": "function",
                    "function": {"name": "resolve", "description": "", "parameters": {"type": "object", "properties": {}}},
                }
            ]

        def dispatch(self, name: str, arguments: dict[str, Any]):
            captured.setdefault("dispatches", []).append((name, arguments))
            return {"ok": True}

    def fake_run_agent(agent_task: dict[str, object], steps: int, tool_specs: list[ChatCompletionToolParam], dispatch_tool_call, system_prompt: str, parent_trace=None, backend_name=None):
        captured["tool_specs"] = tool_specs
        dispatch_tool_call("resolve", {"path": "p"})
        return {"observations": [{"role": "assistant", "content": {"predicted_files": ["src/app.py"], "reasoning": "r"}}]}

    def fake_finalize(task, obs, parent_trace=None):
        finalized["called"] = True
        finalize_calls.append({"task": task, "observations": obs})
        return _runner_result(
            task, ["src/app.py"], reasoning="r", observations=obs, source="finalize"
        )

    monkeypatch.setattr(runner, "_finalize_localization", fake_finalize)

    @contextmanager
    def span_cm():
        class Span:
            id = "span"

            def score(self, name: str, value: float, metadata=None):
                pass

            def update(self, **kwargs):
                pass

            def end(self, **kwargs):
                pass

        yield Span()

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.setattr(runner, "IterativeContextBackend", FakeBackend)
    monkeypatch.setattr(runner, "run_agent", fake_run_agent)
    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())

    result = runner.run_ic_iteration(
        _runner_task(str(repo_dir)),
        score_fn=None,
        steps=1,
        parent_trace=object(),
    )
    assert captured.get("repo") == str(repo_dir)
    assert captured.get("tool_specs")
    assert captured.get("dispatches") == [("resolve", {"path": "p"})]
    assert result.prediction.predicted_files == ["src/app.py"]
    assert finalized["called"] is True
    assert finalize_calls and isinstance(finalize_calls[0]["task"], LCATask)


def test_finalization_uses_json_schema(monkeypatch, tmp_path):
    from contextlib import contextmanager

    calls: list[dict[str, object]] = []

    class DummyCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            # Finalization expects JSON content; return minimal schema-conforming payload on second call.
            content = "{}"
            if len(calls) >= 2:
                content = '{"predicted_files": ["src/app.py"], "reasoning": "r"}'
            response = type(
                "DummyResponse",
                (),
                {
                    "choices": [
                        type("Choice", (), {"message": type("Msg", (), {"content": content})()})()
                    ],
                    "usage": type(
                        "U",
                        (),
                        {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                            "prompt_tokens_details": {},
                            "completion_tokens_details": {},
                        },
                    )(),
                },
            )()
            return response

    class DummyClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": DummyCompletions()})()

    def fake_make_client(model_override=None):
        return DummyClient(), "dummy-model"

    monkeypatch.setattr(runner, "_make_client", fake_make_client)
    monkeypatch.setattr(
        runner,
        "IterativeContextBackend",
        lambda repo, score_fn: type("B", (), {"tool_specs": [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}], "dispatch": lambda self, name, args: {"ok": True}})(),
    )
    @contextmanager
    def span_cm():
        class Span:
            id = "span"

            def update(self, **kwargs): ...
            def end(self, **kwargs): ...

        yield Span()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())

    result = runner.run_ic_iteration(_runner_task(str(tmp_path)), score_fn=None, steps=1, parent_trace=object())
    assert result.prediction.predicted_files == ["src/app.py"]
    assert len(calls) >= 2
    explore_call = calls[0]
    finalize_call = calls[-1]
    assert "tools" in explore_call
    assert explore_call.get("tool_choice") == "required"
    assert "tools" not in finalize_call or finalize_call.get("tools") in (None, [])
    rf = finalize_call.get("response_format")
    assert isinstance(rf, dict)
    assert rf.get("type") == "json_schema"
    js = rf.get("json_schema")
    assert isinstance(js, dict)
    assert js.get("strict") is True
    schema = js.get("schema")
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    assert schema.get("additionalProperties") is False
    assert "predicted_files" in schema.get("properties", {})


def test_finalization_json_schema_uses_provider_supported_subset(monkeypatch, tmp_path):
    from contextlib import contextmanager

    unsupported_keywords = {
        "default",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
    }
    schema_payloads: list[dict[str, object]] = []

    def find_unsupported(value: object, path: str = "$") -> list[str]:
        found: list[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if key in unsupported_keywords:
                    found.append(child_path)
                found.extend(find_unsupported(child, child_path))
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                found.extend(find_unsupported(child, f"{path}[{idx}]"))
        return found

    class DummyCompletions:
        def create(self, **kwargs):
            response_format = kwargs.get("response_format")
            if isinstance(response_format, dict):
                json_schema = response_format.get("json_schema")
                if isinstance(json_schema, dict):
                    schema_payloads.append(json_schema)
                    unsupported = find_unsupported(json_schema)
                    if unsupported:
                        raise AssertionError(
                            f"provider-incompatible schema keywords: {unsupported}"
                        )
            content = '{"predicted_files": ["src/app.py"], "reasoning": "r"}'
            return type(
                "DummyResponse",
                (),
                {
                    "choices": [
                        type("Choice", (), {"message": type("Msg", (), {"content": content})()})()
                    ],
                    "usage": type(
                        "U",
                        (),
                        {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                            "prompt_tokens_details": {},
                            "completion_tokens_details": {},
                        },
                    )(),
                },
            )()

    class DummyClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": DummyCompletions()})()

    monkeypatch.setattr(runner, "_make_client", lambda model_override=None: (DummyClient(), "dummy-model"))
    monkeypatch.setattr(
        runner,
        "IterativeContextBackend",
        lambda repo, score_fn: type(
            "B",
            (),
            {
                "tool_specs": [
                    {
                        "type": "function",
                        "function": {
                            "name": "t",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "dispatch": lambda self, name, args: {"file": "src/app.py"},
            },
        )(),
    )

    @contextmanager
    def span_cm():
        class Span:
            id = "span"

            def update(self, **kwargs): ...
            def end(self, **kwargs): ...

        yield Span()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())

    result = runner.run_ic_iteration(
        _runner_task(str(tmp_path)),
        score_fn=None,
        steps=1,
        parent_trace=object(),
    )

    assert schema_payloads
    assert result.prediction.predicted_files == ["src/app.py"]
    assert result.source == "finalize"


def test_apply_tool_results_preserves_assistant_tool_response_order():
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "resolve", "arguments": "{}"},
    }
    messages = runner._apply_tool_results(  # type: ignore[attr-defined]
        {"identity": {}, "repo": "r", "context": {}},
        [
            {"role": "assistant", "tool_calls": [tool_call], "content": ""},
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"file": "src/app.py"}',
                "result": {"file": "src/app.py"},
            },
        ],
        [],
        "system",
        [],
    )

    assistant_idx = next(
        idx for idx, msg in enumerate(messages) if msg.get("role") == "assistant"
    )
    tool_idx = next(idx for idx, msg in enumerate(messages) if msg.get("role") == "tool")
    assert tool_idx == assistant_idx + 1
    assert messages[tool_idx]["tool_call_id"] == "call-1"


def test_jc_prompt_uses_canonical_tool_names():
    specs: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {"name": "alpha", "description": "", "parameters": {"type": "object", "properties": {}}},
        },
        {
            "type": "function",
            "function": {"name": "beta", "description": "", "parameters": {"type": "object", "properties": {}}},
        },
    ]
    prompt = build_jc_system_prompt(specs)
    assert "- alpha" in prompt
    assert "- beta" in prompt


def test_jc_backend_initializes_github_url(monkeypatch, tmp_path):
    dummy_tool = Tool(name="search_files", description="search", inputSchema={})
    called: dict[str, int] = {"index_repo": 0}

    async def fake_list_tools():
        return [dummy_tool]

    async def fake_call_tool(name: str, arguments: dict[str, Any]):
        if name == "index_repo":
            called["index_repo"] += 1
            assert arguments["url"] == "https://github.com/owner/repo"
            return [TextContent(type="text", text=json.dumps({"repo": "id"}))]
        return []

    monkeypatch.setattr("harness.backends.jc.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.backends.jc.call_tool", fake_call_tool)

    JCodeMunchBackend(repo="https://github.com/owner/repo")
    assert called["index_repo"] == 1


def test_jc_backend_owner_repo_init(monkeypatch):
    dummy_tool = Tool(name="search_files", description="search", inputSchema={})
    called: dict[str, int] = {"index_repo": 0}

    async def fake_list_tools():
        return [dummy_tool]

    async def fake_call_tool(name: str, arguments: dict[str, Any]):
        if name == "index_repo":
            called["index_repo"] += 1
            assert arguments["url"] == "owner/repo"
            return [TextContent(type="text", text=json.dumps({"repo": "id"}))]
        return []

    monkeypatch.setattr("harness.backends.jc.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.backends.jc.call_tool", fake_call_tool)

    JCodeMunchBackend(repo="owner/repo")
    assert called["index_repo"] == 1


def test_jc_backend_rejects_ambiguous_repo(monkeypatch):
    async def fake_list_tools():
        return []

    monkeypatch.setattr("harness.backends.jc.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.backends.jc.call_tool", lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError):
        JCodeMunchBackend(repo="owner/repo/extra")


def test_jc_backend_initializes_local_repo(monkeypatch, tmp_path):
    dummy_tool = Tool(name="search_files", description="search", inputSchema={})
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

    monkeypatch.setattr("harness.backends.jc.list_tools", fake_list_tools)
    monkeypatch.setattr("harness.backends.jc.call_tool", fake_call_tool)

    backend = JCodeMunchBackend(repo=str(tmp_path))
    assert called["resolve"] == 1
    assert called["index"] == 1
    assert backend.tool_specs == [mcp_tool_to_openai_tool(dummy_tool)]
