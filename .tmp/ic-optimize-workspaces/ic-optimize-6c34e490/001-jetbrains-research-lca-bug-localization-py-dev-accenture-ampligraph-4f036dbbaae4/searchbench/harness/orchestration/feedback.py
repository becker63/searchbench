from __future__ import annotations

from pathlib import Path

from harness.orchestration.dependencies import compute_pipeline_diff
from harness.orchestration.types import EvaluationResult, FeedbackPackage
from harness.pipeline.types import PipelineClassification
from harness.prompts import (
    WriterFeedbackFinding,
    WriterGuidance,
    build_writer_optimization_brief,
)
from harness.utils.diff import compute_diff, format_diff, interpret_diff
from harness.utils.test_loader import format_tests_for_prompt, load_tests
from harness.utils.type_loader import build_frontier_context, format_frontier_context


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _writer_guidance_from_evaluation(evaluation: EvaluationResult) -> WriterGuidance:
    if evaluation.error or not evaluation.success:
        return WriterGuidance(
            mode="repair",
            instructions=[
                "Resolve the evaluation failure before attempting score optimization.",
                "Preserve the exact frontier_priority signature and return a float.",
            ],
        )
    regression = evaluation.metrics.get("iteration_regression")
    if regression is True:
        return WriterGuidance(
            mode="repair_regression",
            instructions=[
                "Undo the regression while preserving any validation fixes.",
                "Do not optimize a component by lowering the composed objective.",
            ],
        )
    if evaluation.score_summary and evaluation.score_summary.composed_score is not None:
        return WriterGuidance(
            mode="optimize",
            instructions=[
                "The policy is evaluable; improve the composed score without breaking validation.",
                "Prefer changes that improve weighted visible score components in their stated direction.",
            ],
        )
    return WriterGuidance(
        mode="repair_then_optimize",
        instructions=[
            "First make the policy valid under tests and type checks.",
            "When valid, improve the composed objective while preserving correctness.",
        ],
    )


def build_iteration_feedback(
    evaluation: EvaluationResult,
    prev_score: float | None,
    prev_classified: PipelineClassification | None,
    repo_root: Path,
) -> FeedbackPackage:
    tests = load_tests(repo_root)
    tests_str = format_tests_for_prompt(tests)
    frontier_ctx = build_frontier_context(repo_root)
    frontier_ctx_str = format_frontier_context(frontier_ctx)

    diff_summary = ""
    diff_guidance = ""
    if prev_score is not None and prev_classified is not None:
        try:
            diff = compute_pipeline_diff(
                prev_score,
                _as_float(evaluation.metrics.get("score", 0.0)),
                prev_classified,
                prev_classified,
                compute_diff=compute_diff,
            )
            diff_summary = format_diff(diff)
            diff_guidance = interpret_diff(diff)
        except Exception:
            diff_summary = ""
            diff_guidance = ""

    findings: list[WriterFeedbackFinding] = []
    if evaluation.error:
        findings.append(
            WriterFeedbackFinding(
                kind="evaluation_error",
                message=evaluation.error,
                severity="error",
                source="evaluation",
            )
        )
    for key, value in evaluation.metrics.items():
        normalized_val: str | float | int | bool | None
        if isinstance(value, (str, float, int, bool)) or value is None:
            normalized_val = value
        else:
            normalized_val = str(value)
        severity = (
            "warning"
            if key == "iteration_regression" and value is True
            else "info"
        )
        findings.append(
            WriterFeedbackFinding(
                kind="metric",
                message=f"{key} = {normalized_val}",
                value=normalized_val,
                severity=severity,
                source="evaluation.metrics",
            )
        )

    guidance = _writer_guidance_from_evaluation(evaluation)
    optimization_brief = build_writer_optimization_brief(
        score_summary=evaluation.score_summary,
        findings=findings,
        comparison_summary=evaluation.comparison_summary,
        diff_summary=diff_summary or None,
        diff_guidance=diff_guidance or None,
        guidance=guidance.instructions,
        mode=guidance.mode,
    )

    return FeedbackPackage(
        tests=tests_str,
        frontier_context=frontier_ctx_str,
        frontier_context_details=frontier_ctx,
        optimization_brief=optimization_brief,
    )
