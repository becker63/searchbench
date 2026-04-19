from __future__ import annotations

from harness.scoring.token_usage import aggregate_token_usage_record, extract_token_usage_record


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


def test_extract_preserves_cached_token_breakdown_fields():
    payload = {
        "usage_details": {
            "input": 4,
            "input_cached_tokens": 6,
            "output": 2,
        }
    }
    record = extract_token_usage_record(payload, source="test")
    assert record.available is True
    assert record.usage.fresh_input_tokens == 4
    assert record.usage.cached_input_tokens == 6
    assert record.usage.total_input_tokens == 10
    assert record.usage.output_tokens == 2
    assert record.usage.total_tokens == 12


def test_missing_usage_remains_unavailable():
    payload = {"observations": [{"role": "assistant", "content": "noop"}]}
    record = extract_token_usage_record(payload, source="test")
    assert record.available is False
    assert record.usage is None


def test_aggregates_from_observability_summary_as_full_run_source():
    payload = {
        "raw": {"usage_details": {"input": 10, "output": 2, "total": 12}},
        "observability_summary": {
            "tokens": {
                "fresh_input_tokens": 100,
                "cached_input_tokens": 50,
                "total_input_tokens": 150,
                "output_tokens": 25,
                "total_tokens": 175,
            }
        },
    }

    record = aggregate_token_usage_record(payload, source="test")

    assert record.available is True
    assert record.usage.total_tokens == 175
    assert record.usage.fresh_input_tokens == 100
    assert record.usage.cached_input_tokens == 50
    assert record.usage.total_input_tokens == 150
    assert record.usage.output_tokens == 25


def test_aggregates_agent_and_finalization_usage_without_observation_double_count():
    payload = {
        "observations": [
            {"usage_details": {"input": 1000, "output": 100, "total": 1100}},
        ],
        "raw": {
            "final": {"predicted_files": ["src/app.py"]},
            "usage_details": {"input": 10, "output": 2, "total": 12},
            "raw": {
                "usage_details": {"input": 1000, "output": 100, "total": 1100},
                "observations": [
                    {"usage_details": {"input": 1000, "output": 100, "total": 1100}},
                ],
            },
        },
    }

    first_match = extract_token_usage_record(payload, source="test")
    aggregate = aggregate_token_usage_record(payload, source="test")

    assert first_match.usage.total_tokens == 12
    assert aggregate.available is True
    assert aggregate.usage.total_tokens == 1112
    assert aggregate.usage.prompt_tokens == 1010
    assert aggregate.usage.completion_tokens == 102


def test_aggregates_observation_usage_when_no_mirrored_agent_total_exists():
    payload = {
        "observations": [
            {"usage_details": {"input": 10, "output": 1, "total": 11}},
            {"usage_details": {"input": 20, "output": 2, "total": 22}},
        ]
    }

    record = aggregate_token_usage_record(payload, source="test")

    assert record.available is True
    assert record.usage.total_tokens == 33
    assert record.usage.prompt_tokens == 30
    assert record.usage.completion_tokens == 3
