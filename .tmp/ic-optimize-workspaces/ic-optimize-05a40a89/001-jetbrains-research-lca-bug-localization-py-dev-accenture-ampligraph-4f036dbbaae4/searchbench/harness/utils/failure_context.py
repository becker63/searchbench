from __future__ import annotations

from typing import Mapping, Sequence

from .model_budgets import compute_prompt_char_budget


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"...[truncated {len(text) - limit} chars]"
    return (text[: max(0, limit - len(marker))] + marker)[:limit]


def pack_failure_context(
    pipeline_results: Sequence[Mapping[str, object]],
    classified: Mapping[str, object] | None,
    failed_step: Mapping[str, object] | None,
    writer_error: str | None,
    model_name: str | None,
    repair_attempt: int = 0,
) -> str:
    budget_chars = compute_prompt_char_budget(model_name, repair_attempt=repair_attempt)
    parts: list[str] = []

    if failed_step:
        name = failed_step.get("name") or "<unknown_step>"
        exit_code = failed_step.get("exit_code")
        parts.append(f"Failed step: {name} (exit_code={exit_code})")
        stderr = failed_step.get("stderr")
        stdout = failed_step.get("stdout")
        if isinstance(stderr, str) and stderr.strip():
            parts.append("stderr:\n" + stderr.strip())
        elif isinstance(stdout, str) and stdout.strip():
            parts.append("stdout:\n" + stdout.strip())

    if classified:
        type_err = classified.get("type_errors")
        lint_err = classified.get("lint_errors")
        test_fail = classified.get("test_failures")
        if type_err:
            parts.append(f"type_errors:\n{type_err}")
        if lint_err:
            parts.append(f"lint_errors:\n{lint_err}")
        if test_fail:
            parts.append(f"test_failures:\n{test_fail}")

    if writer_error:
        parts.append(f"writer_error: {writer_error}")

    if not parts and pipeline_results:
        parts.append(str(pipeline_results[0]))

    combined = "\n\n".join(parts)
    return _truncate(combined, budget_chars)
