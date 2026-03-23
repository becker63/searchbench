from __future__ import annotations

from typing import Any, Callable, Iterable, cast

from cerebras.cloud.sdk import Cerebras

_client: Cerebras | None = None


def _get_client() -> Cerebras:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = Cerebras()
    return _client


def _coerce_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(part if isinstance(part, str) else str(part) for part in raw)
    return str(raw)


def generate_policy(feedback: dict[str, Any], current_policy: str) -> str:
    """
    Request an updated policy from Cerebras given feedback metrics and current code.
    """
    client = _get_client()
    messages = [
        {
            "role": "system",
            "content": (
                "You write Python code for harness/policy.py. "
                "Only return valid Python that defines `def score(node, state) -> float:`. "
                "Keep behavior deterministic and side-effect free."
            ),
        },
        {
            "role": "user",
            "content": (
                "Current policy code:\n"
                f"```python\n{current_policy}\n```\n\n"
                "Feedback metrics:\n"
                f"{feedback}\n\n"
                "Update the policy to improve the score."
            ),
        },
    ]

    create = cast(Callable[..., Any], client.chat.completions.create)
    response = create(model="llama3.1-8b", messages=messages)
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Cerebras response contained no choices")

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")

    content = None
    if message is not None:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")

    if content is None and hasattr(first, "text"):
        content = getattr(first, "text")
    if content is None and isinstance(first, dict):
        content = first.get("text")

    if content is None:
        raise ValueError("Cerebras response missing message content")

    return _coerce_content(content)
