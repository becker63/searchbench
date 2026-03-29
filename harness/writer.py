from __future__ import annotations

import time
from typing import Any

from .observability.langfuse import get_tracing_openai_client, record_score, start_span
from .utils.env import get_cerebras_api_key, get_writer_model

_client: Any | None = None


def _get_client():
    global _client  # noqa: PLW0603
    if _client is None:
        api_key = get_cerebras_api_key()
        if not api_key:
            raise RuntimeError("Missing CEREBRAS_API_KEY in environment")
        _client = get_tracing_openai_client(base_url="https://api.cerebras.ai/v1", api_key=api_key)
    return _client, get_writer_model()


def generate_policy(
    feedback: dict[str, Any],
    current_policy: str,
    tests: str,
    scoring_context: str,
    feedback_str: str,
    guidance_hint: str,
    diff_str: str,
    diff_hint: str,
    comparison_summary: str | None = None,
    parent_trace=None,
) -> str:
    """
    Request an updated policy using Cerebras via the OpenAI-compatible client.
    """
    client, model = _get_client()
    if client is None or not hasattr(client, "chat"):
        raise RuntimeError("Writer client is not available")
    writer_obs = start_span(parent_trace, "policy_writer", metadata={"model": model})
    response = None
    record_score(writer_obs, "writer.policy_generated", False)
    record_score(writer_obs, "writer.policy_compiled", False)
    for attempt in range(3):
        attempt_obs = start_span(writer_obs, f"writer_attempt_{attempt}", metadata={"attempt": attempt, "model": model})
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are optimizing a scoring function for a graph search system. "
                            "Output ONLY raw Python. No markdown. No backticks."
                            "\nSTRICT RULES:\n"
                            "- You MUST NOT use import statements\n"
                            "- You MUST NOT reference external modules\n"
                            "- You MUST NOT access the filesystem\n"
                            "- Only use built-in Python operations\n"
                            "Any violation will cause immediate failure."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Current policy:\n{current_policy}\n\n"
                            f"Structured feedback:\n{feedback_str or feedback}\n\n"
                            f"Baseline comparison:\n{comparison_summary or 'N/A'}\n\n"
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
            if attempt_obs and hasattr(attempt_obs, "end"):
                try:
                    attempt_obs.end(metadata={"model": model})
                except Exception:
                    pass
            break
        except Exception as e:  # noqa: BLE001
            if attempt_obs and hasattr(attempt_obs, "end"):
                try:
                    attempt_obs.end(error=e)
                except Exception:
                    pass
            time.sleep(2 * (attempt + 1))
    else:
        if writer_obs and hasattr(writer_obs, "end"):
            try:
                writer_obs.end(error="writer_failed")
            except Exception:
                pass
        raise RuntimeError("Writer failed after retries")

    code = response.choices[0].message.content
    if code is None or "def score" not in code:
        if writer_obs and hasattr(writer_obs, "end"):
            try:
                writer_obs.end(error="missing_score_function")
            except Exception:
                pass
        raise ValueError("Invalid policy: missing score function")

    try:
        compile(code, "<policy>", "exec")
        record_score(writer_obs, "writer.policy_compiled", True, metadata={"model": model})
    except Exception as exc:  # noqa: BLE001
        if writer_obs and hasattr(writer_obs, "end"):
            try:
                writer_obs.end(error=exc, metadata={"model": model})
            except Exception:
                pass
        raise
    record_score(writer_obs, "writer.policy_generated", True, metadata={"model": model})
    if writer_obs and hasattr(writer_obs, "end"):
        try:
            writer_obs.end(metadata={"model": model, "length": len(code), "feedback_keys": list(feedback.keys())})
        except Exception:
            pass
    return code
