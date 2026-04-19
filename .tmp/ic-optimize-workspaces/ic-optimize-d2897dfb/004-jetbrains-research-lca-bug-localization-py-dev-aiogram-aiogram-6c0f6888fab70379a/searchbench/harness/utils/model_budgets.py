from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ModelBudget:
    context_window: int
    tokens_per_minute: int
    requests_per_minute: int
    completion_reserve: int = 2048
    safety_fraction: float = 0.9


_DEFAULT_BUDGET = ModelBudget(
    context_window=32000,
    tokens_per_minute=500000,
    requests_per_minute=500,
    completion_reserve=2048,
    safety_fraction=0.9,
)


_BUDGETS: Mapping[str, ModelBudget] = {
    "gpt-oss-120b": ModelBudget(context_window=131000, tokens_per_minute=1_000_000, requests_per_minute=1000, completion_reserve=4096),
    "llama3.1-8b": ModelBudget(context_window=32768, tokens_per_minute=2_000_000, requests_per_minute=2000, completion_reserve=2048),
    "qwen-3-235b-a22b-instruct-2507": ModelBudget(context_window=131000, tokens_per_minute=500_000, requests_per_minute=500, completion_reserve=4096),
    "zai-glm-4.7": ModelBudget(context_window=131072, tokens_per_minute=500_000, requests_per_minute=500, completion_reserve=4096),
}


def get_model_budget(model: str | None) -> ModelBudget:
    if model and model in _BUDGETS:
        return _BUDGETS[model]
    return _DEFAULT_BUDGET


def compute_prompt_char_budget(model: str | None, repair_attempt: int = 0) -> int:
    budget = get_model_budget(model)
    available_tokens = max(budget.context_window - budget.completion_reserve, 1000)
    # heuristic: later retries shrink prompt modestly to reduce rate/token pressure
    shrink_factor = max(0.5, 1.0 - 0.1 * repair_attempt)
    tokens_for_prompt = int(available_tokens * budget.safety_fraction * shrink_factor)
    # rough char estimate: ~4 chars per token
    return max(tokens_for_prompt * 4, 2000)
