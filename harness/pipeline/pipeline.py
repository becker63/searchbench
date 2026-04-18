"""
Leaf operation: policy pipeline execution (lint/tests/etc.).
No CLI or session policy; receives normalized inputs only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from harness.telemetry.tracing import start_observation
from harness.telemetry.tracing.score_emitter import emit_score_for_handle
from .steps import BasedPyrightStep, PytestStep, RuffFixStep
from .types import PipelineClassification, Step, StepResult


class Pipeline:
    def __init__(self, steps: Iterable[Step], fail_fast: bool = True) -> None:
        self.steps: List[Step] = list(steps)
        self.fail_fast: bool = fail_fast

    def run(self, repo_root: Path, observation: object | None = None, allow_no_parent: bool = False) -> List[StepResult]:
        results: List[StepResult] = []
        if observation is None and not allow_no_parent:
            raise RuntimeError("Parent span required for pipeline tracing; provide observation or set allow_no_parent=True to disable tracing")
        for step in self.steps:
            with start_observation(
                name=step.name,
                parent=observation,
                metadata={"cwd": str(repo_root)},
            ) as step_obs:
                result = step.run(repo_root)
                # Normalize to canonical StepResult to avoid drifting shapes.
                result = result if isinstance(result, StepResult) else StepResult.from_external(result)
                # Enforce success semantics based solely on exit status.
                result.success = result.exit_code == 0
                results.append(result)
                emit_score_for_handle(
                    step_obs,
                    name="exit_code",
                    value=result.exit_code,
                    data_type="NUMERIC",
                    comment=f"step={step.name}",
                )
                emit_score_for_handle(
                    step_obs,
                    name="pipeline_pass",
                    value=result.success,
                    data_type="BOOLEAN",
                    comment=f"step={step.name}",
                )
                terminated = self.fail_fast and result.exit_code != 0
                step_obs.update(
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


def classify_results(results: List[StepResult]) -> PipelineClassification:
    out = PipelineClassification()
    for r in results:
        failed = r.exit_code != 0
        if not failed:
            out.passed.append(r.name)
            continue

        stderr = (r.stderr or "").strip()
        if r.name == "basedpyright":
            out.type_errors = stderr
        elif r.name == "ruff_fix":
            out.lint_errors = stderr
        elif "pytest" in r.name:
            out.test_failures = stderr
    return out


def format_structured_feedback(classified: PipelineClassification, max_chars: int = 4000) -> str:
    parts: list[str] = []

    type_err = classified.type_errors
    lint_err = classified.lint_errors
    test_err = classified.test_failures
    passed = classified.passed

    if type_err:
        parts.append("## TYPE ERRORS (must fix first)\n" + type_err)
    if lint_err:
        parts.append("## LINT ERRORS\n" + lint_err)
    if test_err:
        parts.append("## TEST FAILURES\n" + test_err)
    if passed:
        parts.append("## PASSED STEPS\n" + ", ".join(passed))

    joined = "\n\n".join(parts)
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n\n... truncated"
    return joined


def infer_action_hint(classified: PipelineClassification) -> str:
    if classified.type_errors:
        return "Fix type errors first. Do not attempt logic changes until types pass."
    if classified.lint_errors:
        return "Fix lint/style issues. Ensure valid Python syntax."
    if classified.test_failures:
        return "Tests are failing. Adjust scoring logic to satisfy invariants."
    return "All checks passed. Improve score."
