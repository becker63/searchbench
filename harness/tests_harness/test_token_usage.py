from __future__ import annotations

from harness.localization.token_usage import extract_token_usage_record


def test_extracts_from_top_level_usage_details():
    payload = {
        "usage_details": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "model": "m1",
        }
    }
    record = extract_token_usage_record(payload, source="test")
    assert record.available is True
    assert record.usage.total_tokens == 15
    assert record.usage.prompt_tokens == 10
    assert record.usage.completion_tokens == 5
    assert record.model == "m1"


def test_extracts_from_observations_fallback():
    payload = {
        "observations": [
            {"usage_details": {"input": 7, "output": 3, "total": 10, "model": "m2"}},
        ]
    }
    record = extract_token_usage_record(payload, source="test")
    assert record.available is True
    assert record.usage.total_tokens == 10
    assert record.usage.prompt_tokens == 7
    assert record.usage.completion_tokens == 3
    assert record.model == "m2"


def test_extracts_from_raw_block():
    payload = {
        "raw": {
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            }
        }
    }
    record = extract_token_usage_record(payload, source="test")
    assert record.available is True
    assert record.usage.total_tokens == 3


def test_extracts_from_top_level_usage():
    payload = {
        "usage": {
            "input": 4,
            "output": 6,
            "total": 10,
        }
    }
    record = extract_token_usage_record(payload, source="test")
    assert record.available is True
    assert record.usage.total_tokens == 10


def test_missing_usage_remains_unavailable():
    payload = {"observations": [{"role": "assistant", "content": "noop"}]}
    record = extract_token_usage_record(payload, source="test")
    assert record.available is False
    assert record.usage is None
