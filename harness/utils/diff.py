from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set


@dataclass
class DiffSummary:
    score_delta: float
    fixed_failures: Set[str]
    new_failures: Set[str]
    regression: bool


def extract_failure_keys(classified: Dict[str, object]) -> Set[str]:
    keys: Set[str] = set()
    if isinstance(classified.get("type_errors"), str) and classified["type_errors"]:
        keys.add("type")
    if isinstance(classified.get("lint_errors"), str) and classified["lint_errors"]:
        keys.add("lint")
    if isinstance(classified.get("test_failures"), str) and classified["test_failures"]:
        keys.add("tests")
    return keys


def compute_diff(
    prev_score: float,
    curr_score: float,
    prev_classified: Dict[str, object],
    curr_classified: Dict[str, object],
) -> DiffSummary:
    prev_keys = extract_failure_keys(prev_classified)
    curr_keys = extract_failure_keys(curr_classified)

    fixed = prev_keys - curr_keys
    new = curr_keys - prev_keys

    return DiffSummary(
        score_delta=curr_score - prev_score,
        fixed_failures=fixed,
        new_failures=new,
        regression=bool(new),
    )


def format_diff(diff: DiffSummary) -> str:
    parts: list[str] = []
    parts.append("## CHANGE IMPACT (last iteration)")
    parts.append(f"Score delta: {diff.score_delta:+.4f}")

    if diff.fixed_failures:
        parts.append("Fixed:")
        for item in sorted(diff.fixed_failures):
            parts.append(f"- {item}")

    if diff.new_failures:
        parts.append("New failures introduced:")
        for item in sorted(diff.new_failures):
            parts.append(f"- {item}")

    parts.append(f"Regression: {'yes' if diff.regression else 'no'}")
    return "\n".join(parts)


def interpret_diff(diff: DiffSummary) -> str:
    if diff.new_failures:
        return "Your last change introduced regressions. Be more conservative."
    if diff.fixed_failures and not diff.new_failures:
        return "Good progress. Continue in this direction."
    if diff.score_delta < 0:
        return "Score decreased. Reconsider your heuristic."
    return "No meaningful improvement. Try a different approach."
