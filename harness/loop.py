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
from .utils.env import get_writer_model
from .utils.repo_root import find_repo_root
from .utils.failure_context import pack_failure_context
from .utils.test_loader import format_tests_for_prompt, load_tests
from .utils.type_loader import format_scoring_context, load_scoring_examples, load_scoring_types
from .writer import generate_policy

_POLICY_PATH = Path(__file__).with_name("policy.py")
_MAX_POLICY_REPAIRS = 3
_MAX_FAILURE_SUMMARY_CHARS = 400


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


def _step_succeeded(step: Mapping[str, object]) -> bool:
    if "exit_code" in step and isinstance(step["exit_code"], int):
        return step["exit_code"] == 0
    success_val = step.get("success")
    return bool(success_val)


def _first_failed_step(steps: list[object]) -> Mapping[str, object] | None:
    def _as_mapping(step: object) -> Mapping[str, object] | None:
        if isinstance(step, Mapping):
            return step
        if step is None:
            return None
        data: dict[str, object] = {}
        for key in ("name", "success", "stdout", "stderr", "exit_code"):
            if hasattr(step, key):
                data[key] = getattr(step, key)
        return data if data else None

    for step in steps:
        step_mapping = _as_mapping(step)
        if not step_mapping:
            continue
        if not _step_succeeded(step_mapping):
            return step_mapping
    return None


def _compact_failure(step: Mapping[str, object]) -> str:
    stderr = step.get("stderr")
    stdout = step.get("stdout")
    text = ""
    if isinstance(stderr, str) and stderr.strip():
        text = stderr.strip()
    elif isinstance(stdout, str) and stdout.strip():
        text = stdout.strip()
    return text[:_MAX_FAILURE_SUMMARY_CHARS]


def _map_steps(steps: list[object]) -> list[Mapping[str, object]]:
    mapped: list[Mapping[str, object]] = []
    for step in steps:
        if isinstance(step, Mapping):
            mapped.append(step)
            continue
        data: dict[str, object] = {}
        for key in ("name", "success", "stdout", "stderr", "exit_code"):
            if hasattr(step, key):
                data[key] = getattr(step, key)
        if data:
            mapped.append(data)
    return mapped


def _format_pipeline_failure_context(
    pipeline_results: list[Mapping[str, object]],
    classified: Mapping[str, object] | None,
    writer_error: str | None,
    model_name: str | None,
    repair_attempt: int,
) -> str:
    failed_step = _first_failed_step(pipeline_results)
    return pack_failure_context(
        pipeline_results,
        classified,
        failed_step,
        writer_error,
        model_name,
        repair_attempt=repair_attempt,
    )


def synthesize_valid_policy(
    *,
    repo_root: Path,
    initial_policy: str,
    feedback: dict[str, object],
    tests: str,
    scoring_context: str,
    feedback_str: str,
    guidance_hint: str,
    diff_str: str,
    diff_hint: str,
    comparison_summary: str | None,
    parent_trace=None,
    max_repair_attempts: int = _MAX_POLICY_REPAIRS,
) -> tuple[str, list[Mapping[str, object]], bool, int, dict[str, object]]:
    """
    Generate and iteratively repair policy until validation pipeline passes or attempts exhausted.
    Returns (policy_code, pipeline_results, success, attempts_used, metadata).
    """
    attempts_used = 0
    failure_context_str: str | None = None
    writer_model = get_writer_model()
    pipeline_results: list[Mapping[str, object]] = []
    metadata: dict[str, object] = {}
    last_classified: Mapping[str, object] | None = None
    pipeline = default_pipeline()
    while attempts_used < max_repair_attempts:
        repair_obs = start_span(parent_trace, f"policy_repair_attempt_{attempts_used}", metadata={"attempt": attempts_used})
        guidance = (
            "Repair mode: produce the smallest change that makes harness/policy.py pass ruff, basedpyright, and pytest. "
            "Add missing parameter annotations and return type -> float. Do not optimize score until the policy is valid."
            if attempts_used > 0
            else guidance_hint
        )
        if last_classified and last_classified.get("type_errors"):
            guidance += " Fix all type errors and ensure parameters are annotated; avoid unknown types."
        try:
            new_code = generate_policy(
                feedback={**feedback, **({"pipeline_feedback": last_classified} if last_classified else {})},
                current_policy=initial_policy,
                tests=tests,
                scoring_context=scoring_context,
                feedback_str=str(feedback_str),
                guidance_hint=guidance,
                diff_str=diff_str,
                diff_hint=diff_hint,
                comparison_summary=comparison_summary,
                parent_trace=repair_obs,
                failure_context=failure_context_str,
                repair_attempt=attempts_used,
            )
            new_code = _clean_policy_code(new_code)
            if "import " in new_code:
                if repair_obs and hasattr(repair_obs, "end"):
                    repair_obs.end(error="blocked_import")
                metadata["error"] = "blocked_import"
                break
            _write_policy(new_code)
        except Exception as e:  # noqa: BLE001
            metadata["writer_error"] = str(e)
            attempts_used += 1
            if repair_obs and hasattr(repair_obs, "end"):
                try:
                    repair_obs.end(error=e)
                except Exception:
                    pass
            failure_context_str = _format_pipeline_failure_context([], {}, metadata.get("writer_error"), writer_model, attempts_used)
            continue

        step_results = pipeline.run(repo_root, observation=repair_obs)
        pipeline_results = _map_steps(list(step_results))
        normalized_results = pipeline_results or _map_steps(list(step_results))
        pipeline_success = all(_step_succeeded(r) for r in normalized_results)
        attempts_used += 1
        if pipeline_success:
            # Ruff may have modified files; accept on-disk version.
            final_policy = _read_policy()
            metadata["repair_attempts"] = attempts_used - 1
            metadata["max_policy_repairs"] = max_repair_attempts
            metadata["accepted_policy_source"] = "post_pipeline_disk"
            metadata["accepted_policy_changed_by_pipeline"] = final_policy != new_code
            if repair_obs and hasattr(repair_obs, "end"):
                repair_obs.end(metadata={"status": "passed"})
            return final_policy, pipeline_results, True, attempts_used, metadata

        classified = classify_results(step_results)
        last_classified = classified
        failed_step = _first_failed_step(pipeline_results)
        if failed_step:
            metadata["failed_step"] = failed_step.get("name")
            metadata["failed_exit_code"] = failed_step.get("exit_code")
            metadata["failed_summary"] = _compact_failure(failed_step)
        metadata["pipeline_feedback"] = classified
        metadata["error"] = "pipeline_failed"
        failure_context_str = _format_pipeline_failure_context(pipeline_results, classified, metadata.get("writer_error"), writer_model, attempts_used)
        initial_policy = _read_policy()
        if repair_obs and hasattr(repair_obs, "end"):
            repair_obs.end(metadata={"status": "failed", "error": metadata.get("error")})
    metadata.setdefault("repair_attempts", attempts_used)
    metadata["max_policy_repairs"] = max_repair_attempts
    if pipeline_results and all(_step_succeeded(r) for r in pipeline_results):
        return _read_policy(), pipeline_results, True, attempts_used, metadata
    return initial_policy, pipeline_results, False, attempts_used, metadata


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
    resolved_repo_path = repo_path if isinstance(repo_path, str) else None
    jc_repo_id: str | None = None
    root_meta = {"task": dict(task), "iterations": iterations}
    writer_model = get_writer_model()
    run_trace = start_span(parent_trace, "run_loop", metadata=root_meta, input=task) if parent_trace else start_trace("run_loop", metadata=root_meta, input=task)
    index_obs = None
    if isinstance(repo_path, str):
        try:
            index_obs = start_span(run_trace, "index_repo", metadata={"repo": repo_path})
            result = index_folder(repo_path)
            repo_id = result["repo"]
            jc_repo_id = repo_id
            if index_obs and hasattr(index_obs, "end"):
                index_obs.end(metadata={"repo": repo_id})
        except Exception as e:  # noqa: BLE001
            if index_obs is not None:
                try:
                    index_obs.end(error=e)
                except Exception:
                    pass

    try:
        for i in range(iterations):
            ic_task = dict(task)
            if resolved_repo_path is not None:
                ic_task["repo"] = resolved_repo_path
            jc_task = dict(task)
            if jc_repo_id is not None:
                jc_task["repo"] = jc_repo_id
            elif resolved_repo_path is not None:
                jc_task["repo"] = resolved_repo_path

            iteration_span = start_span(
                run_trace,
                f"iteration_{i}",
                metadata={"iteration": i, "iterations": iterations, "task": dict(ic_task), "jc_repo": jc_task.get("repo")},
            )
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
                if iteration_span and hasattr(iteration_span, "end"):
                    try:
                        iteration_span.end(metadata={"status": "failed", "error": metrics_map.get("error")})
                    except Exception:
                        pass
                continue

            ic_result = run_ic_iteration(ic_task, score_fn, parent_trace=iteration_span)
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
            initial_policy = _read_policy()
            candidate_policy, pipeline_results, pipeline_success, repairs_used, repair_meta = synthesize_valid_policy(
                repo_root=repo_root,
                initial_policy=initial_policy,
                feedback=metrics_map,
                tests=tests_str,
                scoring_context=scoring_ctx,
                feedback_str=str(metrics_map.get("error", "")),
                guidance_hint="Fix errors before improving score",
                diff_str="",
                diff_hint="",
                comparison_summary=comparison_summary,
                parent_trace=iteration_span,
                max_repair_attempts=_MAX_POLICY_REPAIRS,
            )
            metrics_map.update(repair_meta)
            metrics_map["pipeline_passed"] = pipeline_success
            metrics_map["repair_attempts"] = repair_meta.get("repair_attempts", repairs_used)
            metrics_map["max_policy_repairs"] = repair_meta.get("max_policy_repairs", _MAX_POLICY_REPAIRS)
            metrics_map["pipeline_feedback"] = repair_meta.get("pipeline_feedback", {})
            if pipeline_results:
                metrics_map["failed_step"] = metrics_map.get("failed_step") or _first_failed_step(pipeline_results).get("name") if _first_failed_step(pipeline_results) else None
                metrics_map["failed_exit_code"] = metrics_map.get("failed_exit_code") or (metrics_map.get("failed_step") and repair_meta.get("failed_exit_code"))
                metrics_map["failed_summary"] = metrics_map.get("failed_summary") or repair_meta.get("failed_summary")
            if not pipeline_success:
                record_score(iteration_span, "pipeline_passed", False)
                history.append(metrics_map)
                if iteration_span and hasattr(iteration_span, "end"):
                    try:
                        iteration_span.end(metadata={"status": "failed", "pipeline_passed": False, "error": metrics_map.get("error"), "writer_error": metrics_map.get("writer_error")})
                    except Exception:
                        pass
                continue

            record_score(iteration_span, "pipeline_passed", True)
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
