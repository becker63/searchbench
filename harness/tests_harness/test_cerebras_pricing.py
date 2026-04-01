from harness.observability.cerebras_pricing import CEREBRAS_PRICING, pricing_payload


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
    assert "match_pattern" in first and "prices" in first
    assert set(first["prices"].keys()) >= {"input", "output"}
