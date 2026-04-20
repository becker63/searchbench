from __future__ import annotations

CEREBRAS_PRICING: list[dict[str, str | float]] = [
    {
        "name": "Zaizai GLM 4.7",
        "match_pattern": r"(?i)zaizai.*glm.*4\.7",
        "input_price_per_million": 2.25,
        "output_price_per_million": 2.75,
    },
    {
        "name": "GPT OSS 120B",
        "match_pattern": r"(?i)gpt[-_ ]?oss[-_ ]?120b",
        "input_price_per_million": 0.35,
        "output_price_per_million": 0.75,
    },
    {
        "name": "Llama 3.1 8B",
        "match_pattern": r"(?i)llama[-_ ]?3\.1[-_ ]?8b",
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
                "model_name": entry["name"],
                "match_pattern": entry["match_pattern"],
                # Use pricing tiers (default tier only) to align with v4 Models API
                "pricing_tiers": [
                    {
                        "name": "standard",
                        "is_default": True,
                        "priority": 0,
                        "conditions": [],
                        "prices": {
                            "input": entry["input_price_per_million"],
                            "output": entry["output_price_per_million"],
                        },
                    }
                ],
                "unit": "TOKENS",
            }
        )
    return payload


def cost_details_for_usage(model: str | None, usage_details: dict[str, object] | None) -> dict[str, float] | None:
    """
    Derive cost_details locally using the Cerebras pricing table to guarantee costs when inference lags.
    Prices are per million tokens; usage is expected to be token counts.
    """
    if not model or not usage_details:
        return None
    price_entry = None
    import re

    for entry in CEREBRAS_PRICING:
        pattern = entry["match_pattern"]
        if isinstance(pattern, str) and re.search(pattern, model, flags=re.IGNORECASE):
            price_entry = entry
            break
    if not price_entry:
        return None
    def _as_num(val: object) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    inp = _as_num(usage_details.get("input") or usage_details.get("prompt_tokens") or usage_details.get("total"))
    out = _as_num(usage_details.get("output") or usage_details.get("completion_tokens") or 0)
    input_raw = price_entry.get("input_price_per_million")
    output_raw = price_entry.get("output_price_per_million")
    input_price = float(input_raw) / 1_000_000 if isinstance(input_raw, (int, float)) else 0.0
    output_price = float(output_raw) / 1_000_000 if isinstance(output_raw, (int, float)) else 0.0
    costs: dict[str, float] = {}
    if inp > 0:
        costs["input"] = inp * input_price
    if out > 0:
        costs["output"] = out * output_price
    total = sum(costs.values())
    if total > 0:
        costs["total"] = total
    return costs or None


def sync_cerebras_models(client: object | None = None) -> None:
    """
    Sync Cerebras model definitions to Langfuse Cloud Models API.

    Requires a v4 Langfuse client with `api.models.create` available.
    """
    if client is None:
        from . import get_langfuse_client

        client = get_langfuse_client()
    models_api = getattr(getattr(client, "api", None), "models", None)
    create_fn = getattr(models_api, "create", None) if models_api else None
    list_fn = getattr(models_api, "list", None) if models_api else None
    delete_fn = getattr(models_api, "delete", None) if models_api else None
    if not callable(create_fn):
        raise RuntimeError("Langfuse client does not expose api.models.create; sync models via HTTP or UI.")
    if not callable(list_fn) or not callable(delete_fn):
        raise RuntimeError("Langfuse client does not expose api.models.list/delete; cannot upsert models.")

    # Upsert: fetch existing model ids by name, delete, then recreate with current patterns/prices.
    existing_by_name: dict[str, str | None] = {}
    try:
        page = 1
        while True:
            res = list_fn(page=page, limit=100)
            for model in getattr(res, "data", []) or []:
                name_val = getattr(model, "model_name", None)
                model_id_val = getattr(model, "id", None)
                if isinstance(name_val, str):
                    existing_by_name[name_val] = str(model_id_val) if model_id_val is not None else None
            meta = getattr(res, "meta", None)
            if not meta or getattr(meta, "page", 1) >= getattr(meta, "total_pages", 1):
                break
            page += 1
    except Exception:
        existing_by_name = {}

    for model_def in pricing_payload():
        name_val = model_def.get("model_name")
        name = str(name_val) if name_val is not None else ""
        existing_id = existing_by_name.get(name)
        if existing_id:
            try:
                delete_fn(id=str(existing_id))
            except Exception:
                # best-effort; continue to create if delete fails
                pass
        create_fn(**model_def)


__all__ = ["CEREBRAS_PRICING", "pricing_payload", "sync_cerebras_models"]
