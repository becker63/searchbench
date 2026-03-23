from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .utils.env import load_env

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client  # noqa: PLW0603
    if _client is None:
        load_env()
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise RuntimeError("Missing CEREBRAS_API_KEY in environment")
        _client = OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=api_key,
        )
        model = os.environ.get("WRITER_MODEL", "gpt-oss-120b")
        print(f"[LLM] Writer model: {model}")
    return _client


def generate_policy(
    feedback: dict[str, Any],
    current_policy: str,
    tests: str,
    scoring_context: str,
    feedback_str: str,
    guidance_hint: str,
    diff_str: str,
    diff_hint: str,
) -> str:
    """
    Request an updated policy using Cerebras via the OpenAI-compatible client.
    """
    client = _get_client()
    model = os.environ.get("WRITER_MODEL", "gpt-oss-120b")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are optimizing a scoring function for a graph search system. Output ONLY valid Python.",
            },
            {
                "role": "user",
                "content": (
                    f"Current policy:\n{current_policy}\n\n"
                    f"Structured feedback:\n{feedback_str or feedback}\n\n"
                    f"Guidance:\n{guidance_hint}\n\n"
                    f"{diff_str}\n\n"
                    f"Diff guidance:\n{diff_hint}\n\n"
                    f"Tests (source of truth, do not modify):\n{tests}\n\n"
                    f"Scoring interface + examples (STRICT CONTRACT):\n{scoring_context}\n\n"
                    "Instructions:\n"
                    "- Fix errors in order of priority:\n"
                    "  1. TYPE ERRORS\n"
                    "  2. LINT ERRORS\n"
                    "  3. TEST FAILURES\n\n"
                    "- Do NOT attempt to optimize until all tests pass\n"
                    "- MUST define: def score(node, state) -> float\n"
                    "- MUST be deterministic\n"
                    "- MUST NOT import external libraries\n"
                    "- MUST operate only on provided node/state data\n\n"
                    "Return ONLY valid Python code implementing the score function.\n"
                ),
            },
        ],
    )

    code = response.choices[0].message.content
    if code is None or "def score" not in code:
        raise ValueError("Invalid policy: missing score function")

    compile(code, "<policy>", "exec")
    return code
