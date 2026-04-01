from __future__ import annotations

CEREBRAS_PRICING: list[dict[str, str | float]] = [
    {
        "name": "Zaizai GLM 4.7",
        "match_pattern": r"(?i)zaizai.*glm.*4\\.7",
        "input_price_per_million": 2.25,
        "output_price_per_million": 2.75,
    },
    {
        "name": "GPT OSS 120B",
        "match_pattern": r"(?i)gpt[-_ ]oss[-_ ]120b",
        "input_price_per_million": 0.35,
        "output_price_per_million": 0.75,
    },
    {
        "name": "Llama 3.1 8B",
        "match_pattern": r"(?i)llama[-_ ]3\\.1[-_ ]8b",
        "input_price_per_million": 0.10,
        "output_price_per_million": 0.10,
    },
    {
        "name": "Qwen 3 235B Instruct",
        "match_pattern": r"(?i)qwen[-_ ]3[-_ ]235b[-_ ]instruct",
        "input_price_per_million": 0.60,
        "output_price_per_million": 1.20,
    },
]


def pricing_payload() -> list[dict[str, object]]:
    """Return a Langfuse Models API payload for Cerebras pricing."""
    payload: list[dict[str, object]] = []
    for entry in CEREBRAS_PRICING:
        payload.append(
            {
                "name": entry["name"],
                "match_pattern": entry["match_pattern"],
                "prices": {
                    "input": entry["input_price_per_million"],
                    "output": entry["output_price_per_million"],
                },
            }
        )
    return payload


__all__ = ["CEREBRAS_PRICING", "pricing_payload"]
