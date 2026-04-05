from __future__ import annotations

from contextlib import contextmanager
import pytest

from harness.localization.models import LCAContext, LCATaskIdentity
from harness.loop import runner_agent as runner
from harness.loop import agent_common
from harness import writer


class DummySpan:
    def __init__(self, name: str = "obs"):
        self.name = name
        self.ended: list[dict[str, object]] = []
        self.scores: list[tuple[str, float]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def start_observation(self, **kwargs):
        return self

    def end(self, **kwargs):
        self.ended.append(kwargs)

    def score(self, name: str, value: float | int | bool, metadata=None):
        self.scores.append((name, float(value)))

    def update(self, **kwargs):
        pass


def _agent_payload(repo: str = "r") -> dict[str, object]:
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    context = LCAContext(issue_title="bug", issue_body="details")
    return {"identity": identity.model_dump(), "context": context.model_dump(), "repo": repo}


def test_run_ic_iteration_emits_scores(monkeypatch):
    scores: list[tuple[str, float]] = []
    monkeypatch.setattr(
        runner,
        "emit_score_for_handle",
        lambda handle, **kwargs: scores.append((kwargs.get("name"), float(kwargs.get("value", 0.0)))),
    )
    @contextmanager
    def span_cm(name="span"):
        yield DummySpan(name)

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(runner, "run_agent", lambda *a, **k: {"observations": [{"tool": "t", "result": {"graph": {"nodes": [1, 2]}}}]})
    monkeypatch.setattr(
        runner,
        "IterativeContextBackend",
        lambda repo, score_fn: type("B", (), {"tool_specs": [], "dispatch": lambda self, name, args: {}})(),
    )
    monkeypatch.setattr(
        runner,
        "_finalize_localization",
        lambda task, obs, parent_trace=None: runner.LocalizationRunnerResult(
            predicted_files=["src/app.py"], observations=obs, source="finalize"
        ),
    )

    parent = object()
    task_payload = _agent_payload("r")
    result = runner.run_ic_iteration(task_payload, score_fn=lambda n, s, c=None: 1.0, steps=1, parent_trace=parent)
    assert result["node_count"] == 2
    assert ("ic.node_count", 2.0) in scores


def test_writer_records_policy_events(monkeypatch):
    attempt_scores: list[tuple[str, float]] = []

    class DummyMessage:
        def __init__(self):
            self.content = '{"code": "def score(node, state):\\n    return 1.0\\n"}'

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

    def fake_client():
        return DummyClient(), "test-model"

    monkeypatch.setattr(writer, "_get_client", fake_client)
    @contextmanager
    def span_cm(name="writer"):
        yield DummySpan(name)

    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(
        writer,
        "emit_score_for_handle",
        lambda obs, **kwargs: attempt_scores.append((kwargs.get("name"), float(kwargs.get("value", 0.0)))),
    )

    result = writer.generate_policy(
        feedback={"a": 1},
        current_policy="def score(node, state):\n    return 0.0\n",
        tests="",
        scoring_context="",
        feedback_str="",
        guidance_hint="",
        diff_str="",
        diff_hint="",
        comparison_summary="IC vs JC summary",
        parent_trace=DummySpan("parent"),
    )
    assert "def score" in result
    assert ("writer.policy_compiled", 1.0) in attempt_scores
    assert ("writer.policy_generated", 1.0) in attempt_scores


def test_run_agent_truncates_tool_content(monkeypatch):
    class DummyResponse:
        class Choice:
            class Msg:
                def __init__(self):
                    self.tool_calls = [
                        type("Call", (), {"id": "1", "function": type("F", (), {"name": "tool", "arguments": "{}"})()})()
                    ]

            message = Msg()

        choices = [Choice()]
        usage = type(
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

        def __init__(self):
            self.usage = self.__class__.usage
        usage = type(
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

        def __init__(self):
            self.usage = self.__class__.usage
        usage = type(
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

        def __init__(self):
            self.usage = self.__class__.usage
        usage = type(
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
        usage = type(
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
        usage = type(
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
        usage = type(
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
        usage = type(
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
        usage = type(
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
        usage = type(
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
        def __init__(self):
            self.calls: list[list[dict[str, object]]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs["messages"])
            return DummyResponse()

    completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": completions})()

    monkeypatch.setattr(runner, "_make_client", lambda: (DummyClient(), "model"))
    @contextmanager
    def span_cm():
        yield type("S", (), {"id": "span", "end": lambda self, **kw: None, "update": lambda self, **kw: None})()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())
    long_result = "X" * (runner._MAX_TOOL_CONTENT_CHARS * 2)
    observations = runner.run_agent(
        _agent_payload(),
        steps=2,
        tool_specs=[{"type": "function", "function": {"name": "tool", "description": "", "parameters": {}}}],
        dispatch_tool_call=lambda *a, **k: long_result,
        system_prompt="sys",
        parent_trace=object(),
        backend_name="test",
    )
    assert observations["observations"]
    assert len(completions.calls) == 2
    second_call_msgs = completions.calls[1]
    tool_msg = next(msg for msg in second_call_msgs if msg.get("role") == "tool")
    assert len(str(tool_msg.get("content", ""))) <= runner._MAX_TOOL_CONTENT_CHARS


def test_run_agent_retries_on_context_error(monkeypatch):
    class ContextError(Exception):
        status_code = 400

    class DummyResponse:
        usage = type(
            "U",
            (),
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "prompt_tokens_details": {},
                "completion_tokens_details": {},
            },
        )()

        class Choice:
            class Msg:
                def __init__(self):
                    self.tool_calls = [
                        type("Call", (), {"id": "1", "function": type("F", (), {"name": "tool", "arguments": "{}"})()})()
                    ]

            message = Msg()

        choices = [Choice()]

    class DummyCompletions:
        def __init__(self):
            self.calls: list[list[dict[str, object]]] = []
            self.failed = False

        def create(self, **kwargs):
            self.calls.append(kwargs["messages"])
            if not self.failed:
                self.failed = True
                raise ContextError("Please reduce the length")
            return DummyResponse()

    completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": completions})()

    monkeypatch.setattr(runner, "_make_client", lambda: (DummyClient(), "model"))
    patched_usage = lambda resp: agent_common.UsageEnvelope.model_validate({"input": 1, "output": 1, "total": 2})
    monkeypatch.setattr(agent_common, "usage_from_response", patched_usage)
    monkeypatch.setitem(runner.run_agent.__globals__, "usage_from_response", patched_usage)
    @contextmanager
    def span_cm():
        yield type("S", (), {"id": "span", "end": lambda self, **kw: None, "update": lambda self, **kw: None})()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())
    result = runner.run_agent(
        _agent_payload(),
        steps=1,
        tool_specs=[{"type": "function", "function": {"name": "tool", "description": "", "parameters": {}}}],
        dispatch_tool_call=lambda *a, **k: {},
        system_prompt="sys",
        parent_trace=object(),
        backend_name="test",
    )
    assert len(completions.calls) == 2
    assert result["observations"]


def test_usage_from_response_maps_openai_fields():
    class DummyUsage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15
        prompt_tokens_details = {"cached_tokens": 2}
        completion_tokens_details = {"reasoning_tokens": 3}

    class DummyResponse:
        usage = DummyUsage()

    mapped = agent_common.usage_from_response(DummyResponse())
    assert mapped is not None
    flat = mapped.model_dump_flat()
    assert flat["input"] == 10
    assert flat["output"] == 5
    assert flat["total"] == 15
    assert flat["prompt_tokens_details.cached_tokens"] == 2
    assert flat["input_cached_tokens"] == 2
    assert flat["completion_tokens_details.reasoning_tokens"] == 3
    assert flat["output_reasoning_tokens"] == 3


def test_run_agent_requires_usage(monkeypatch):
    class DummyResponse:
        class Choice:
            class Msg:
                def __init__(self):
                    self.tool_calls = [
                        type("Call", (), {"id": "1", "function": type("F", (), {"name": "tool", "arguments": "{}"})()})()
                    ]

            message = Msg()

        choices = [Choice()]

    class DummyCompletions:
        def create(self, **kwargs):
            return DummyResponse()

    class DummyClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": DummyCompletions()})()

    monkeypatch.setattr(runner, "_make_client", lambda: (DummyClient(), "model"))
    monkeypatch.setattr(agent_common, "usage_from_response", lambda resp: None)
    monkeypatch.setitem(runner.run_agent.__globals__, "usage_from_response", lambda resp: None)

    @contextmanager
    def span_cm():
        yield type("S", (), {"id": "span", "end": lambda self, **kw: None, "update": lambda self, **kw: None})()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())

    with pytest.raises(RuntimeError, match="missing usage"):
        runner.run_agent(
            _agent_payload(),
            steps=1,
            tool_specs=[{"type": "function", "function": {"name": "tool", "description": "", "parameters": {}}}],
            dispatch_tool_call=lambda *a, **k: {},
            system_prompt="sys",
            parent_trace=object(),
            backend_name="test",
        )
