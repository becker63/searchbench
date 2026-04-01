from __future__ import annotations

import json

import pytest

from typing import TypedDict

from harness import writer


class DummySpan:
    def __init__(self):
        self.ended: list[dict[str, object]] = []
        self.scores: list[tuple[str, float]] = []

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
    scoring_context: str
    feedback_str: str
    guidance_hint: str
    diff_str: str
    diff_hint: str
    comparison_summary: str | None
    parent_trace: object | None
    failure_context: str | None
    repair_attempt: int


def _base_inputs() -> BaseInputs:
    return {
        "feedback": {"a": 1},
        "current_policy": "def score(node, state):\n    return 0.0\n",
        "tests": "",
        "scoring_context": "",
        "feedback_str": "",
        "guidance_hint": "",
        "diff_str": "",
        "diff_hint": "",
        "comparison_summary": "IC vs JC summary",
        "parent_trace": None,
    }


def test_generate_policy_accepts_structured_code(monkeypatch):
    scores: list[tuple[str, float]] = []
    calls: list[dict[str, object]] = []
    payload = {"code": "def score(node: object, state: object, context: object | None = None) -> float:\n    return 1.0\n"}
    monkeypatch.setattr(writer, "_get_client", lambda: (_make_client(json.dumps(payload), calls), "model"))
    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: DummySpan())
    monkeypatch.setattr(
        writer,
        "emit_score_for_handle",
        lambda obs, **kwargs: scores.append((kwargs.get("name"), float(kwargs.get("value", 0.0)))),
    )
    code = writer.generate_policy(**_base_inputs())
    assert "def score" in code
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
    payload = {"code": "```python\ndef score():\n    pass\n```"}
    monkeypatch.setattr(writer, "_get_client", lambda: (_make_client(json.dumps(payload), []), "model"))
    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: DummySpan())
    monkeypatch.setattr(writer, "emit_score_for_handle", lambda *a, **k: None)
    with pytest.raises(ValueError):
        writer.generate_policy(**_base_inputs())


def test_generate_policy_rejects_invalid_python(monkeypatch):
    payload = {"code": "def score(:\n    pass"}
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
    parsed = {"code": "def score(node, state):\n    return 2.0\n"}
    monkeypatch.setattr(writer, "_get_client", lambda: (_make_client(None, calls, parsed=parsed), "model"))
    monkeypatch.setattr(writer, "start_observation", lambda *a, **k: DummySpan())
    monkeypatch.setattr(writer, "emit_score_for_handle", lambda *a, **k: None)
    code = writer.generate_policy(**_base_inputs())
    assert "return 2.0" in code
