from __future__ import annotations

import difflib
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jcodemunch_mcp.tools.index_folder import index_folder

from .pipeline import (
    classify_results,
    default_pipeline,
)
from .policy_loader import load_policy
from .runner import run_ic_iteration, run_jc_iteration
from .scorer import score
from .utils.debug import print_iteration_debug
from .utils.diff import compute_diff, format_diff, interpret_diff
from .utils.repo_root import find_repo_root
from .utils.test_loader import format_tests_for_prompt, load_tests
from .utils.type_loader import (
    format_scoring_context,
    load_scoring_examples,
    load_scoring_types,
)
from .writer import generate_policy

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


def print_policy_diff(old: str, new: str) -> None:
    if old == new:
        print("[POLICY] No change")
        return

    print("[POLICY] Updated")
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        lineterm="",
    )
    lines = list(diff)
    max_lines = 50
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print("... diff truncated")


def _clean_policy_code(code: str) -> str:
    if "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if code.startswith("python"):
                code = code[len("python") :].lstrip()
    return code.strip()


def _hash_tests_dir(repo_root: Path) -> str:
    tests_path = repo_root / "iterative-context" / "tests"
    hasher = hashlib.sha256()
    if not tests_path.exists():
        return hasher.hexdigest()
    for file in sorted(tests_path.rglob("*.py")):
        hasher.update(str(file.relative_to(repo_root)).encode())
        hasher.update(file.read_bytes())
    return hasher.hexdigest()


def run_loop(
    task: Mapping[str, object], iterations: int = 5
) -> list[dict[str, object]]:
    """
    Closed-loop optimizer: execute, score, rewrite policy, repeat.
    """
    history: list[dict[str, object]] = []
    prev_score: float | None = None
    prev_classified: dict[str, object] | None = None
    repo_path = task.get("repo")
    if isinstance(repo_path, str):
        print(f"[INDEX] Indexing repo: {repo_path}")
        try:
            result = index_folder(repo_path)
            repo_id = result["repo"]
            print(f"[INDEX] Repo registered as: {repo_id}")
            task = dict(task)
            task["repo"] = repo_id
        except Exception as e:  # noqa: BLE001
            print(f"[INDEX ERROR] {e}")

    try:
        for i in range(iterations):
            try:
                score_fn = load_policy()
            except ImportError as e:
                metrics_map = {
                    "score": -10.0,
                    "error": f"policy_import_error: {str(e)}",
                }
                print("[POLICY ERROR]", metrics_map["error"])
                history.append(metrics_map)
                continue
            print(f"[TASK] {task}")
            ic_result = run_ic_iteration(dict(task), score_fn)
            jc_result = run_jc_iteration(dict(task))
            combined_result = {"iterative_context": ic_result, "jcodemunch": jc_result}
            metrics_map: dict[str, object] = dict(score(combined_result))
            if ic_result.get("node_count", 0) == 0:
                metrics_map["score"] = float(metrics_map.get("score", 0.0)) - 1.0
                metrics_map["error"] = "no_tool_usage"
            print(f"=== ITERATION {i} ===")
            print("[IC RESULT]")
            print_iteration_debug(i, {"iterative_context": ic_result, "jcodemunch": {}}, metrics_map)
            print("[JC RESULT]")
            print_iteration_debug(i, {"iterative_context": {}, "jcodemunch": jc_result}, metrics_map)
            print("[METRICS]")
            print(metrics_map)

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
                    feedback_str=metrics_map.get("error", ""),
                    guidance_hint="Fix errors before improving score",
                    diff_str="",
                    diff_hint="",
                )
                new_code = _clean_policy_code(new_code)
                if "import " in new_code:
                    print("[POLICY BLOCKED] contains import")
                    history.append({"error": "blocked_import"})
                    continue
                print_policy_diff(current_code, new_code)
                print("\n[POLICY FULL OUTPUT]\n")
                print(new_code)
                print("\n[END POLICY]\n")
                _write_policy(new_code)
            except Exception as e:  # noqa: BLE001
                print(f"[POLICY ERROR] {e}")
                print("[POLICY] Reusing previous policy")
                history.append(metrics_map)
                continue

            tests_hash_after = _hash_tests_dir(repo_root)
            if tests_hash_after != tests_hash_before:
                _write_policy(current_code)
                print("[SECURITY] Tests modified — reverting policy")
                history.append({"error": "test_integrity_modified"})
                continue

            pipeline = default_pipeline()
            results = pipeline.run(repo_root)
            if any(not r.success for r in results):
                classified = classify_results(results)
                feedback_str = metrics_map.get("error", "")
                guidance_hint = "Fix errors before improving score"
                current_score = _as_float(metrics_map.get("score", 0.0))
                diff_str = ""
                diff_hint = ""
                if prev_score is not None and prev_classified is not None:
                    diff = compute_diff(
                        prev_score, current_score, prev_classified, classified
                    )
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
                except Exception as e:  # noqa: BLE001
                    print(f"[POLICY ERROR] {e}")
                continue

            history.append(metrics_map)
            prev_score = _as_float(metrics_map.get("score", 0.0))
            prev_classified = {
                "type_errors": "",
                "lint_errors": "",
                "test_failures": "",
                "passed": [],
            }
    except KeyboardInterrupt:
        return history

    return history
