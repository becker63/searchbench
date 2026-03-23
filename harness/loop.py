from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any
from pathlib import Path

from .policy_loader import load_policy
from .runner import run_both
from .scorer import score
from .writer import generate_policy
from .utils.repo_root import find_repo_root
from .utils.test_loader import format_tests_for_prompt, load_tests
from .utils.type_loader import (
    format_scoring_context,
    load_scoring_examples,
    load_scoring_types,
)
from .pipeline import (
    default_pipeline,
    classify_results,
    format_structured_feedback,
    infer_action_hint,
)
from .utils.diff import compute_diff, format_diff, interpret_diff

_POLICY_PATH = Path(__file__).with_name("policy.py")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_policy(path: Path = _POLICY_PATH) -> str:
    return path.read_text(encoding="utf-8")


def _write_policy(code: str, path: Path = _POLICY_PATH) -> None:
    if path.name != "policy.py":
        raise RuntimeError("Unauthorized write attempt")
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(code, encoding="utf-8")
    tmp_path.replace(path)


def _hash_tests_dir(repo_root: Path) -> str:
    tests_path = repo_root / "iterative-context" / "tests"
    hasher = hashlib.sha256()
    if not tests_path.exists():
        return hasher.hexdigest()
    for file in sorted(tests_path.rglob("*.py")):
        hasher.update(str(file.relative_to(repo_root)).encode())
        hasher.update(file.read_bytes())
    return hasher.hexdigest()


def run_loop(task: Mapping[str, object], iterations: int = 5) -> list[dict[str, object]]:
    """
    Closed-loop optimizer: execute, score, rewrite policy, repeat.
    """
    history: list[dict[str, object]] = []
    prev_score: float | None = None
    prev_classified: dict[str, object] | None = None

    for _ in range(iterations):
        score_fn = load_policy()
        result = run_both(dict(task), score_fn)
        metrics_map: dict[str, object] = dict(score(result))

        repo_root = find_repo_root()
        tests = load_tests(repo_root)
        tests_str = format_tests_for_prompt(tests)
        types_str = load_scoring_types(repo_root)
        examples_str = load_scoring_examples(repo_root)
        scoring_ctx = format_scoring_context(types_str, examples_str)
        tests_hash_before = _hash_tests_dir(repo_root)

        current_code = _read_policy()
        try:
            new_code = generate_policy(
                feedback=metrics_map,
                current_policy=current_code,
                tests=tests_str,
                scoring_context=scoring_ctx,
                feedback_str="",
                guidance_hint="",
                diff_str="",
                diff_hint="",
            )
            _write_policy(new_code)
        except Exception:
            history.append(metrics_map)
            continue

        tests_hash_after = _hash_tests_dir(repo_root)
        if tests_hash_after != tests_hash_before:
            _write_policy(current_code)
            history.append({"error": "test_integrity_modified"})
            continue

        pipeline = default_pipeline()
        results = pipeline.run(repo_root)
        if any(not r.success for r in results):
            classified = classify_results(results)
            feedback_str = format_structured_feedback(classified)
            guidance_hint = infer_action_hint(classified)
            current_score = _as_float(metrics_map.get("score", 0.0))
            diff_str = ""
            diff_hint = ""
            if prev_score is not None and prev_classified is not None:
                diff = compute_diff(prev_score, current_score, prev_classified, classified)
                diff_str = format_diff(diff)
                diff_hint = interpret_diff(diff)
            prev_score = current_score
            prev_classified = classified
            try:
                updated_code = generate_policy(
                    feedback={**metrics_map, "pipeline_feedback": classified},
                    current_policy=_read_policy(),
                    tests=tests_str,
                    scoring_context=scoring_ctx,
                    feedback_str=feedback_str,
                    guidance_hint=guidance_hint,
                    diff_str=diff_str,
                    diff_hint=diff_hint,
                )
                _write_policy(updated_code)
            except Exception:
                pass
            continue

        history.append(metrics_map)
        prev_score = _as_float(metrics_map.get("score", 0.0))
        prev_classified = {"type_errors": "", "lint_errors": "", "test_failures": "", "passed": []}

    return history
