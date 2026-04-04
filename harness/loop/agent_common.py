from __future__ import annotations

"""
Shared agent utilities that are reused by runner and writer agents.
"""

from collections.abc import Mapping
from typing import Any


def usage_from_response(resp: Any) -> dict[str, object] | None:
    """
    Normalize OpenAI-style usage into Langfuse usage_details shape.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    payload: dict[str, object] = {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    if isinstance(prompt_tokens, (int, float)):
        payload["prompt_tokens"] = prompt_tokens
        payload["input"] = prompt_tokens
    if isinstance(completion_tokens, (int, float)):
        payload["completion_tokens"] = completion_tokens
        payload["output"] = completion_tokens
    if isinstance(total_tokens, (int, float)):
        payload["total_tokens"] = total_tokens
        payload["total"] = total_tokens
    if isinstance(prompt_details, Mapping):
        for key, val in prompt_details.items():
            if isinstance(val, (int, float)):
                payload[f"prompt_tokens_details.{key}"] = val
                payload[f"input_{key}"] = val
    if isinstance(completion_details, Mapping):
        for key, val in completion_details.items():
            if isinstance(val, (int, float)):
                payload[f"completion_tokens_details.{key}"] = val
                payload[f"output_{key}"] = val
    return payload or None


__all__ = ["usage_from_response"]
