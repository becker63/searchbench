from __future__ import annotations

import json

import pytest

from typing import TypedDict

from harness.agents import writer
from harness.prompts import WriterOptimizationBrief, build_writer_optimization_brief
from harness.utils.type_loader import FrontierContext


class DummySpan:
    def __init__(self):
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


class _DummyMessage:
    def __init__(self, content, parsed=None):
        self.content = content
        self.parsed = parsed


class _DummyChoice:
    def __init__(self, content, parsed=None):
        self.message = _DummyMessage(content, parsed)


class _DummyResponse:
    def __init__(self, content, parsed=None):
        self.choices = [_DummyChoice(content, parsed)]
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


def _make_client(content, capture, parsed=None):
    class DummyCompletions:
        def create(self, **kwargs):
            capture.append(kwargs)
            return _DummyResponse(content, parsed)

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = DummyChat()

    return DummyClient()


class BaseInputs(TypedDict, total=False):
    feedback: dict[str, object]
    current_policy: str
    tests: str
    frontier_context: str
    frontier_context_details: FrontierContext
    optimization_brief: WriterOptimizationBrief
    parent_trace: object | None
    failure_context: str | None
    repair_attempt: int


def _base_inputs() -> BaseInputs:
    return {
        "feedback": {"a": 1},
        "current_policy": "def frontier_priority(node: \"GraphNode\", graph: \"Graph\", step: int):\n    return 0.0\n",
        "tests": "",
        "frontier_context": "",
        "frontier_context_details": FrontierContext(
            signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
            graph_models="",
            types="",
            examples="",
            notes="",
        ),
        "optimization_brief": build_writer_optimization_brief(comparison_summary="IC vs JC summary"),
        "parent_trace": DummySpan(),
    }


def test_generate_policy_accepts_structured_code(monkeypatch):
    scores: list[tuple[str, float]] = []
    calls: list[dict[str, object]] = []
    payload = {"code": "def frontier_priority(node: \"GraphNode\", graph: \"Graph\", step: int) -> float:\n    return 1.0\n"}
    monkeypatch.setattr(writer, "_get_client", lambda: (_make_client(json.dumps(payload), calls), "model"))
    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: DummySpan())
    monkeypatch.setattr(
        writer,
        "emit_score_for_handle",
        lambda obs, **kwargs: scores.append((kwargs.get("name"), float(kwargs.get("value", 0.0)))),
    )
    code = writer.generate_policy(**_base_inputs())
    assert "def frontier_priority" in code
    assert ("writer.policy_compiled", 1.0) in scores
    assert ("writer.policy_generated", 1.0) in scores
    assert calls
    assert isinstance(calls[0]["response_format"], dict)
    response_format = calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert isinstance(response_format["json_schema"], dict)
    json_schema = response_format["json_schema"]
    assert isinstance(json_schema["schema"], dict)
    schema = json_schema["schema"]
    assert schema["required"] == ["code"]
    assert schema["properties"] == {"code": {"type": "string"}}
    assert response_format["json_schema"]["strict"] is True
    assert schema.get("additionalProperties") is False
    max_tokens = calls[0]["max_completion_tokens"]
    assert isinstance(max_tokens, int)
    assert max_tokens > 0


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing code
        {"code": None},
        {"code": 123},
    ],
)
def test_generate_policy_rejects_missing_or_nonstring_code(monkeypatch, payload):
    monkeypatch.setattr(writer, "_get_client", lambda: (_make_client(json.dumps(payload), []), "model"))
    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: DummySpan())
    monkeypatch.setattr(writer, "emit_score_for_handle", lambda *a, **k: None)
    with pytest.raises(ValueError):
        writer.generate_policy(**_base_inputs())


def test_generate_policy_rejects_fenced_code(monkeypatch):
    payload = {"code": "```python\ndef frontier_priority():\n    pass\n```"}
    monkeypatch.setattr(writer, "_get_client", lambda: (_make_client(json.dumps(payload), []), "model"))
    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: DummySpan())
    monkeypatch.setattr(writer, "emit_score_for_handle", lambda *a, **k: None)
    with pytest.raises(ValueError):
        writer.generate_policy(**_base_inputs())


def test_generate_policy_rejects_invalid_python(monkeypatch):
    payload = {"code": "def frontier_priority(:\n    pass"}
    monkeypatch.setattr(writer, "_get_client", lambda: (_make_client(json.dumps(payload), []), "model"))
    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: DummySpan())
    monkeypatch.setattr(writer, "emit_score_for_handle", lambda *a, **k: None)
    with pytest.raises(SyntaxError):
        writer.generate_policy(**_base_inputs())


def test_generate_policy_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(writer, "_get_client", lambda: (_make_client("not-json", []), "model"))
    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: DummySpan())
    monkeypatch.setattr(writer, "emit_score_for_handle", lambda *a, **k: None)
    with pytest.raises(ValueError):
        writer.generate_policy(**_base_inputs())


def test_generate_policy_accepts_parsed_fallback(monkeypatch):
    calls: list[dict[str, object]] = []
    parsed = {"code": "def frontier_priority(node: \"GraphNode\", graph: \"Graph\", step: int):\n    return 2.0\n"}
    monkeypatch.setattr(writer, "_get_client", lambda: (_make_client(None, calls, parsed=parsed), "model"))
    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: DummySpan())
    monkeypatch.setattr(writer, "emit_score_for_handle", lambda *a, **k: None)
    code = writer.generate_policy(**_base_inputs())
    assert "return 2.0" in code


def test_frontier_context_is_derived_from_structured_context(monkeypatch):
    derived: list[str] = []

    def fake_format(ctx):
        derived.append(ctx.signature)
        return "rendered-context"

    payload = {"code": "def frontier_priority(node: \"GraphNode\", graph: \"Graph\", step: int) -> float:\n    return 1.0\n"}
    monkeypatch.setattr(writer, "_get_client", lambda: (_make_client(json.dumps(payload), []), "model"))
    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: DummySpan())
    monkeypatch.setattr(writer, "emit_score_for_handle", lambda *a, **k: None)
    monkeypatch.setattr(writer, "format_frontier_context", fake_format)
    inputs = _base_inputs()
    inputs["frontier_context"] = ""  # force derivation from structured context
    inputs["frontier_context_details"] = FrontierContext(
        signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
        graph_models="",
        types="",
        examples="from iterative_context.graph_models import Graph, GraphNode\n"
        "def frontier_priority(node: GraphNode, graph: Graph, step: int) -> float:\n    return 1.0\n",
        notes="",
    )
    writer.generate_policy(**inputs)
    assert derived and derived[0] == "SelectionCallable = Callable[[GraphNode, Graph, int], float]"
