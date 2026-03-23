from __future__ import annotations

import os
from typing import Any, Callable, cast

from openai import OpenAI

from .utils.env import load_env
from .tools.normalize import normalize_iterative_context, normalize_jcodemunch
from .tools.parser import execute_tool_call
from .tools.registry import get_tool_specs


def _require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"task must include a non-empty string for '{key}'")
    return value


def _make_client() -> OpenAI:
    load_env()
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing CEREBRAS_API_KEY in environment")
    model = os.environ.get("RUNNER_MODEL", "llama3.1-8b")
    client = OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
    )
    print(f"[LLM] Runner model: {model}")
    return client


def run_agent(task: dict[str, object], score_fn: Any = None, steps: int = 5) -> dict[str, object]:
    """
    Run a simple tool-using agent for a fixed number of steps.
    """
    client = _make_client()
    tool_specs = get_tool_specs()

    messages: list[Any] = [
        {
            "role": "system",
            "content": "You are a code exploration agent. Use tools to gather information.",
        },
        {
            "role": "user",
            "content": f"Task:\n{task}\n\nUse tools to explore.",
        },
    ]

    observations: list[dict[str, object]] = []

    model = os.environ.get("RUNNER_MODEL", "llama3.1-8b")

    for _ in range(steps):
        response = client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            tools=cast(Any, tool_specs),
        )

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            break

        for call in tool_calls:
            parsed: dict[str, Any] | None = None
            try:
                parsed = {
                    "tool": call.function.name,
                    "arguments": call.function.arguments or {},
                }
                result = execute_tool_call(parsed, score_fn=score_fn)
                observations.append({"tool": parsed["tool"], "result": result})
                messages.append({"role": "tool", "content": str(result)})
            except Exception as exc:  # noqa: BLE001
                tool_name = parsed["tool"] if parsed else "unknown"
                observations.append({"tool": tool_name, "error": str(exc)})
                messages.append({"role": "tool", "content": f"ERROR: {exc}"})

    return {"observations": observations}


def run_ic_agent(task: dict[str, object], score_fn: Any, steps: int = 5) -> dict[str, object]:
    return run_agent(task, score_fn=score_fn, steps=steps)


def run_jc_agent(task: dict[str, object], steps: int = 5) -> dict[str, object]:
    return run_agent(task, score_fn=None, steps=steps)


def run_both(task: dict[str, object], score_fn: Callable[[object, object], float]):
    symbol = _require_str(task, "symbol")
    repo = _require_str(task, "repo")
    symbol_id = str(task.get("symbol_id", symbol))
    agent_task: dict[str, object] = {"symbol": symbol, "repo": repo, "symbol_id": symbol_id}

    ic_raw = run_ic_agent(agent_task, score_fn=score_fn)
    jc_raw = run_jc_agent(agent_task)

    ic = normalize_iterative_context(ic_raw)
    jc = normalize_jcodemunch(jc_raw)

    return {
        "iterative_context": ic,
        "jcodemunch": jc,
    }
