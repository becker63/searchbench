from __future__ import annotations

import difflib
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jcodemunch_mcp.tools.index_folder import index_folder

from .observability.langfuse import flush_langfuse, record_score, start_span, start_trace
from .observability.baselines import baseline_metrics_from_jc
from .pipeline import classify_results, default_pipeline
from .policy_loader import load_policy
from .runner import run_ic_iteration
from .scorer import score
from .utils.comparison import build_comparison, format_comparison_summary
from .utils.diff import compute_diff, format_diff, interpret_diff
from .utils.repo_root import find_repo_root
from .utils.test_loader import format_tests_for_prompt, load_tests
from .utils.type_loader import format_scoring_context, load_scoring_examples, load_scoring_types
from .writer import generate_policy

_POLICY_PATH = Path(__file__).with_name("policy.py")


def _as_float(value: Any) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
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


def _tool_call_stats(bucket: Mapping[str, object] | None) -> tuple[int, int]:
    if not isinstance(bucket, Mapping):
        return 0, 0
    observations = bucket.get("observations")
    if not isinstance(observations, list):
        return 0, 0
    errors = 0
    for obs in observations:
        if isinstance(obs, Mapping) and "error" in obs:
            errors += 1
    return len(observations), errors


def run_loop(
    task: Mapping[str, object],
    iterations: int = 5,
    parent_trace=None,
    baseline_snapshot: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """
    Closed-loop optimizer: execute, score, rewrite policy, repeat.
    """
    history: list[dict[str, object]] = []
    prev_score: float | None = None
    prev_classified: dict[str, object] | None = None
    repo_path = task.get("repo")
    root_meta = {"task": dict(task), "iterations": iterations}
    run_trace = start_span(parent_trace, "run_loop", metadata=root_meta, input=task) if parent_trace else start_trace("run_loop", metadata=root_meta, input=task)
    index_obs = None
    if isinstance(repo_path, str):
        try:
            index_obs = start_span(run_trace, "index_repo", metadata={"repo": repo_path})
            result = index_folder(repo_path)
            repo_id = result["repo"]
            if index_obs and hasattr(index_obs, "end"):
                index_obs.end(metadata={"repo": repo_id})
            task = dict(task)
            task["repo"] = repo_id
        except Exception as e:  # noqa: BLE001
            if index_obs is not None:
                try:
                    index_obs.end(error=e)
                except Exception:
                    pass

    try:
        for i in range(iterations):
            iteration_span = start_span(run_trace, f"iteration_{i}", metadata={"iteration": i, "iterations": iterations, "task": dict(task)})
            policy_load_obs = None
            try:
                policy_load_obs = start_span(iteration_span, "policy_load", metadata={"iteration": i})
                score_fn = load_policy()
                if policy_load_obs and hasattr(policy_load_obs, "end"):
                    policy_load_obs.end(metadata={"status": "loaded"})
            except ImportError as e:
                if policy_load_obs is not None:
                    try:
                        policy_load_obs.end(error=e)
                    except Exception:
                        pass
                metrics_map = {
                    "score": -10.0,
                    "error": f"policy_import_error: {str(e)}",
                }
                record_score(iteration_span, "metrics.score", _as_float(metrics_map.get("score", 0.0)))
                history.append(metrics_map)
                continue

            ic_result = run_ic_iteration(dict(task), score_fn, parent_trace=iteration_span)
            jc_result = baseline_snapshot.get("jc_result") if isinstance(baseline_snapshot, Mapping) else {}
            jc_metrics = baseline_snapshot.get("jc_metrics") if isinstance(baseline_snapshot, Mapping) else {}
            if not isinstance(jc_metrics, Mapping):
                jc_metrics = baseline_metrics_from_jc(jc_result if isinstance(jc_result, Mapping) else {})
            combined_result = {"iterative_context": ic_result, "jcodemunch": jc_result}
            metrics_map: dict[str, object] = dict(score(combined_result))
            if ic_result.get("node_count", 0) == 0:
                metrics_map["score"] = _as_float(metrics_map.get("score", 0.0)) - 1.0
                metrics_map["error"] = "no_tool_usage"
            total_ic, errors_ic = _tool_call_stats(ic_result)
            total_jc, errors_jc = _tool_call_stats(jc_result if isinstance(jc_result, Mapping) else {})
            total_calls = total_ic + total_jc
            total_errors = errors_ic + errors_jc
            metrics_map["tool_error_rate"] = (total_errors / total_calls) if total_calls else 0.0
            metrics_map["no_tool_usage"] = total_ic == 0
            comparison = build_comparison(metrics_map, jc_metrics if isinstance(jc_metrics, Mapping) else {})
            comparison_summary = format_comparison_summary(comparison)
            metrics_map.setdefault("iteration_regression", False)
            metrics_map["jc_baseline_metrics"] = dict(jc_metrics) if isinstance(jc_metrics, Mapping) else {}
            metrics_map["comparison"] = comparison
            for name, value in metrics_map.items():
                if isinstance(value, (int, float, bool)):
                    record_score(iteration_span, f"metrics.{name}", float(value))

            repo_root = find_repo_root()
            tests = load_tests(repo_root)
            tests_str = format_tests_for_prompt(tests)
            types_str = load_scoring_types(repo_root)
            examples_str = load_scoring_examples(repo_root)
            scoring_ctx = format_scoring_context(types_str, examples_str)
            tests_hash_before = _hash_tests_dir(repo_root)

            current_code = _read_policy()
            policy_gen_obs = start_span(iteration_span, "policy_generation", metadata={"iteration": i})
            try:
                new_code = generate_policy(
                    feedback=metrics_map,
                    current_policy=current_code,
                    tests=tests_str,
                    scoring_context=scoring_ctx,
                    feedback_str=str(metrics_map.get("error", "")),
                    guidance_hint="Fix errors before improving score",
                    diff_str="",
                    diff_hint="",
                    comparison_summary=comparison_summary,
                    parent_trace=policy_gen_obs,
                )
                new_code = _clean_policy_code(new_code)
                if "import " in new_code:
                    if policy_gen_obs and hasattr(policy_gen_obs, "end"):
                        policy_gen_obs.end(error="blocked_import")
                    history.append({"error": "blocked_import"})
                    continue
                if policy_gen_obs and hasattr(policy_gen_obs, "end"):
                    policy_gen_obs.end(metadata={"status": "generated", "length": len(new_code)})
                _write_policy(new_code)
            except Exception as e:  # noqa: BLE001
                if policy_gen_obs and hasattr(policy_gen_obs, "end"):
                    try:
                        policy_gen_obs.end(error=e)
                    except Exception:
                        pass
                history.append(metrics_map)
                continue

            integrity_obs = start_span(iteration_span, "test_integrity", metadata={"iteration": i})
            tests_hash_after = _hash_tests_dir(repo_root)
            if tests_hash_after != tests_hash_before:
                _write_policy(current_code)
                if integrity_obs and hasattr(integrity_obs, "end"):
                    integrity_obs.end(error="test_integrity_modified")
                history.append({"error": "test_integrity_modified"})
                continue
            if integrity_obs and hasattr(integrity_obs, "end"):
                integrity_obs.end(metadata={"status": "ok"})

            pipeline = default_pipeline()
            pipeline_results = pipeline.run(repo_root, observation=iteration_span)
            pipeline_obs = start_span(iteration_span, "pipeline", metadata={"iteration": i})
            if any(not r.success for r in pipeline_results):
                classified = classify_results(pipeline_results)
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
                    metrics_map["iteration_regression"] = diff.regression
                    record_score(iteration_span, "iteration_regression", diff.regression)
                prev_score = current_score
                prev_classified = classified
                try:
                    updated_code = generate_policy(
                        feedback={**metrics_map, "pipeline_feedback": classified},
                        current_policy=_read_policy(),
                        tests=tests_str,
                        scoring_context=scoring_ctx,
                        feedback_str=str(feedback_str),
                        guidance_hint=guidance_hint,
                        diff_str=diff_str,
                        diff_hint=diff_hint,
                        comparison_summary=comparison_summary,
                        parent_trace=iteration_span,
                    )
                    _write_policy(updated_code)
                    metrics_map["pipeline_passed"] = False
                    record_score(iteration_span, "pipeline_passed", False)
                except Exception:
                    metrics_map["pipeline_passed"] = False
                    record_score(iteration_span, "pipeline_passed", False)
                if pipeline_obs and hasattr(pipeline_obs, "end"):
                    pipeline_obs.end(metadata={"status": "failed"})
                continue

            metrics_map["pipeline_passed"] = True
            record_score(iteration_span, "pipeline_passed", True)
            if pipeline_obs and hasattr(pipeline_obs, "end"):
                pipeline_obs.end(metadata={"status": "passed"})
            history.append(metrics_map)
            prev_score = _as_float(metrics_map.get("score", 0.0))
            prev_classified = {
                "type_errors": "",
                "lint_errors": "",
                "test_failures": "",
                "passed": [],
            }
            if iteration_span and hasattr(iteration_span, "end"):
                try:
                    iteration_span.end(metadata={"status": "complete", "score": prev_score})
                except Exception:
                    pass
    except KeyboardInterrupt:
        if run_trace and hasattr(run_trace, "end"):
            try:
                run_trace.end(error="interrupted")
            except Exception:
                pass
    finally:
        if run_trace and hasattr(run_trace, "end"):
            try:
                run_trace.end(metadata={"iterations_completed": len(history)})
            except Exception:
                pass
        flush_langfuse()
    return history
