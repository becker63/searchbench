from harness.telemetry.tracing.cerebras_pricing import (
    CEREBRAS_PRICING,
    cost_details_for_usage,
    pricing_payload,
)


def test_cerebras_pricing_entries_present():
    names = {entry["name"] for entry in CEREBRAS_PRICING}
    assert {
        "Zaizai GLM 4.7",
        "GPT OSS 120B",
        "Llama 3.1 8B",
        "Qwen 3 235B Instruct",
    }.issubset(names)
    for entry in CEREBRAS_PRICING:
        assert "match_pattern" in entry
        assert "input_price_per_million" in entry
        assert "output_price_per_million" in entry


def test_pricing_payload_shapes():
    payload = pricing_payload()
    assert len(payload) >= 4
    first = payload[0]
    assert "match_pattern" in first and "pricing_tiers" in first
    tier = first["pricing_tiers"][0]
    assert set(tier["prices"].keys()) >= {"input", "output"}


def test_cost_details_for_usage_matches_llama():
    usage = {"input": 1000, "output": 500}  # tokens
    costs = cost_details_for_usage("llama3.1-8b", usage)
    assert costs is not None
    # 0.10 per million tokens => 0.0001 per thousand
    assert round(costs["input"], 6) == round(1000 * 0.10 / 1_000_000, 6)
    assert round(costs["output"], 6) == round(500 * 0.10 / 1_000_000, 6)
    assert round(costs["total"], 6) == round(costs["input"] + costs["output"], 6)


def test_cost_details_for_usage_no_match_returns_none():
    usage = {"input": 1000, "output": 500}
    assert cost_details_for_usage("unknown-model", usage) is None
