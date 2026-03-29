from __future__ import annotations

from typing import Iterable, List, Dict
from pathlib import Path

from ..observability.langfuse import start_span, record_score
from .steps import BasedPyrightStep, PytestStep, RuffFixStep
from .types import Step, StepResult


class Pipeline:
    def __init__(self, steps: Iterable[Step], fail_fast: bool = True) -> None:
        self.steps: List[Step] = list(steps)
        self.fail_fast: bool = fail_fast

    def run(self, repo_root: Path, observation=None) -> List[StepResult]:
        results: List[StepResult] = []
        for step in self.steps:
            step_obs = start_span(observation, name=step.name, metadata={"cwd": str(repo_root)})
            result = step.run(repo_root)
            results.append(result)
            record_score(step_obs, "exit_code", result.exit_code, metadata={"step": step.name})
            record_score(step_obs, "pipeline_pass", result.success, metadata={"step": step.name})
            terminated = self.fail_fast and not result.success
            if step_obs and hasattr(step_obs, "end"):
                step_obs.end(
                    metadata={
                        "step": step.name,
                        "success": result.success,
                        "exit_code": result.exit_code,
                        "stdout_excerpt": (result.stdout or "")[:400],
                        "stderr_excerpt": (result.stderr or "")[:400],
                        "fail_fast": self.fail_fast,
                        "terminated_pipeline": terminated,
                    }
                )
            if terminated:
                break
        return results


def default_pipeline() -> Pipeline:
    return Pipeline([RuffFixStep(), BasedPyrightStep(), PytestStep()])


def summarize_results(results: List[StepResult]) -> str:
    lines: list[str] = []
    for result in results:
        status = "passed" if result.success else "FAILED"
        lines.append(f"[{result.name}] {status}")
        if not result.success and result.stderr:
            lines.append(result.stderr.strip())
            lines.append("")
    summary = "\n".join(lines).strip()
    if len(summary) > 2000:
        summary = summary[:2000] + "\n... (truncated)"
    return summary


def classify_results(results: List[StepResult]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "type_errors": "",
        "lint_errors": "",
        "test_failures": "",
        "passed": [],
    }
    for r in results:
        if r.success:
            passed_list = out["passed"]
            if isinstance(passed_list, list):
                passed_list.append(r.name)
            continue

        stderr = (r.stderr or "").strip()
        if r.name == "basedpyright":
            out["type_errors"] = stderr
        elif r.name == "ruff_fix":
            out["lint_errors"] = stderr
        elif "pytest" in r.name:
            out["test_failures"] = stderr
    return out


def format_structured_feedback(classified: Dict[str, object], max_chars: int = 4000) -> str:
    parts: list[str] = []

    type_err = classified.get("type_errors")
    lint_err = classified.get("lint_errors")
    test_err = classified.get("test_failures")
    passed = classified.get("passed")

    if isinstance(type_err, str) and type_err:
        parts.append("## TYPE ERRORS (must fix first)\n" + type_err)
    if isinstance(lint_err, str) and lint_err:
        parts.append("## LINT ERRORS\n" + lint_err)
    if isinstance(test_err, str) and test_err:
        parts.append("## TEST FAILURES\n" + test_err)
    if isinstance(passed, list) and passed:
        parts.append("## PASSED STEPS\n" + ", ".join(passed))

    joined = "\n\n".join(parts)
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n\n... truncated"
    return joined


def infer_action_hint(classified: Dict[str, object]) -> str:
    if isinstance(classified.get("type_errors"), str) and classified.get("type_errors"):
        return "Fix type errors first. Do not attempt logic changes until types pass."
    if isinstance(classified.get("lint_errors"), str) and classified.get("lint_errors"):
        return "Fix lint/style issues. Ensure valid Python syntax."
    if isinstance(classified.get("test_failures"), str) and classified.get("test_failures"):
        return "Tests are failing. Adjust scoring logic to satisfy invariants."
    return "All checks passed. Improve score."
