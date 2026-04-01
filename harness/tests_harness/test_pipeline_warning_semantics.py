from __future__ import annotations

from harness.pipeline.pipeline import classify_results, format_structured_feedback, infer_action_hint
from harness.pipeline.types import PipelineClassification, StepResult


def test_basedpyright_warning_not_classified_as_failure():
    results = [
        StepResult(
            name="basedpyright",
            success=True,
            stdout="warning: reportUnusedParameter",
            stderr="warning: partially unknown",
            exit_code=0,
        )
    ]
    classified = classify_results(results)
    assert isinstance(classified, PipelineClassification)
    assert classified.type_errors == ""
    feedback = format_structured_feedback(classified)
    assert "TYPE ERRORS" not in feedback
    assert infer_action_hint(classified) == "All checks passed. Improve score."


def test_pytest_warning_not_classified_as_failure():
    results = [
        StepResult(
            name="pytest_iterative_context",
            success=True,
            stdout="collected 10 items, 1 warning",
            stderr="warning about deprecation",
            exit_code=0,
        )
    ]
    classified = classify_results(results)
    assert isinstance(classified, PipelineClassification)
    assert classified.test_failures == ""
    feedback = format_structured_feedback(classified)
    assert "TEST FAILURES" not in feedback
    assert infer_action_hint(classified) == "All checks passed. Improve score."


def test_ruff_warning_not_classified_as_failure():
    results = [
        StepResult(
            name="ruff_fix",
            success=True,
            stdout="warning lint notice",
            stderr="",
            exit_code=0,
        )
    ]
    classified = classify_results(results)
    assert isinstance(classified, PipelineClassification)
    assert classified.lint_errors == ""
    feedback = format_structured_feedback(classified)
    assert "LINT ERRORS" not in feedback
    assert infer_action_hint(classified) == "All checks passed. Improve score."
