from __future__ import annotations

from pathlib import Path

import pytest

from harness.localization.models import (
    LCAContext,
    LCAGold,
    LCATask,
    LCATaskIdentity,
    LocalizationPrediction,
)
from harness.localization.runtime.evaluate import (
    LocalizationEvaluationResult,
    LocalizationEvaluationTaskResult,
)
from harness.orchestration import evaluation, feedback, writer
from harness.orchestration.types import (
    EvaluationMetrics,
    EvaluationResult,
    ICResult,
    JCResult,
)
from harness.pipeline.types import PipelineClassification
from harness.prompts import WriterOptimizationBrief, build_writer_optimization_brief
from harness.scoring import ScoreBundle, ScoreContext, ScoreEngine, summarize_batch_scores
from harness.utils.type_loader import FrontierContext


def _identity() -> LCATaskIdentity:
    return LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )


def _task(tmp_path: Path) -> LCATask:
    return LCATask(
        identity=_identity(),
        context=LCAContext(issue_title="bug", issue_body="body"),
        gold=LCAGold(changed_files=["a.py"]),
        repo=str(tmp_path),
    )


def _score_bundle() -> ScoreBundle:
    return ScoreEngine().evaluate(
        ScoreContext(
            predicted_files=("a.py",),
            gold_files=("a.py",),
            gold_min_hops=1,
            issue_min_hops=1,
        )
    )


def test_evaluation_module_builds_loop_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task = _task(tmp_path)
    bundle = _score_bundle()
    task_result = LocalizationEvaluationTaskResult(
        task=task,
        score_bundle=bundle,
        prediction=LocalizationPrediction(predicted_files=["a.py"]),
    )
    summary = summarize_batch_scores({"task": bundle})

    def evaluate_batch(**kwargs: object) -> LocalizationEvaluationResult:
        assert kwargs["tasks"]
        return LocalizationEvaluationResult(
            score_summary=summary,
            machine_score=summary.aggregate_composed_score,
            items=[task_result],
            failure=None,
        )

    monkeypatch.setattr(evaluation, "evaluate_localization_batch", evaluate_batch)
    monkeypatch.setattr(evaluation, "_read_policy", lambda: "policy")

    result = evaluation.evaluate_policy_on_item(task, None, None, 0)

    assert result.policy_code == "policy"
    assert result.control_score == pytest.approx(summary.aggregate_composed_score)
    assert result.metrics.get("score") == pytest.approx(bundle.composed_score)
    assert result.ic_result.get("predicted_files") == ["a.py"]


def test_feedback_module_builds_writer_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frontier = FrontierContext(signature="sig", graph_models="", types="", examples="", notes="")
    evaluation_result = EvaluationResult(
        metrics=EvaluationMetrics.model_validate(
            {
                "score": 0.25,
                "iteration_regression": True,
                "additional_metrics": [{"name": "gold_hop", "value": 0.5}],
            }
        ),
        ic_result=ICResult(),
        jc_result=JCResult(),
        jc_metrics=EvaluationMetrics(),
        comparison_summary="baseline comparison",
        policy_code="policy",
        error="runner failed",
        success=False,
    )

    monkeypatch.setattr(feedback, "load_tests", lambda repo_root: [("test.py", "def test_ok(): pass")])
    monkeypatch.setattr(feedback, "format_tests_for_prompt", lambda tests: "tests")
    monkeypatch.setattr(feedback, "build_frontier_context", lambda repo_root: frontier)
    monkeypatch.setattr(feedback, "format_frontier_context", lambda ctx: "frontier")

    package = feedback.build_iteration_feedback(
        evaluation_result,
        None,
        None,
        tmp_path,
    )

    assert package.tests == "tests"
    assert package.frontier_context == "frontier"
    assert package.frontier_context_details == frontier
    assert package.optimization_brief.guidance.mode == "repair"
    assert package.optimization_brief.comparison.summary == "baseline comparison"
    assert any(
        finding.message == "runner failed"
        for finding in package.optimization_brief.findings
    )


def test_writer_module_normalizes_success_and_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    frontier = FrontierContext(signature="", graph_models="", types="", examples="", notes="")
    brief = build_writer_optimization_brief(guidance=["hint"])

    def generator(
        current_policy: str,
        tests: str,
        frontier_context: str,
        frontier_context_details: FrontierContext,
        optimization_brief: WriterOptimizationBrief,
        feedback: dict[str, object] | None = None,
        parent_trace: object | None = None,
        failure_context: str | None = None,
        repair_attempt: int = 0,
    ) -> str:
        assert feedback == {
            "pipeline_feedback": PipelineClassification(type_errors="bad").model_dump()
        }
        return "```python\ndef frontier_priority(node, graph, step):\n    return 1.0\n```"

    monkeypatch.setattr(writer, "generate_policy", generator)
    code, error = writer.attempt_policy_generation(
        initial_policy="old",
        tests="tests",
        frontier_context="frontier",
        frontier_context_details=frontier,
        optimization_brief=brief,
        failure_context_str=None,
        repair_attempt=0,
        pipeline_feedback=PipelineClassification(type_errors="bad"),
    )

    assert error is None
    assert code == "def frontier_priority(node, graph, step):\n    return 1.0"

    def failing_generator(
        current_policy: str,
        tests: str,
        frontier_context: str,
        frontier_context_details: FrontierContext,
        optimization_brief: WriterOptimizationBrief,
        feedback: dict[str, object] | None = None,
        parent_trace: object | None = None,
        failure_context: str | None = None,
        repair_attempt: int = 0,
    ) -> str:
        raise RuntimeError("writer broke")

    monkeypatch.setattr(writer, "generate_policy", failing_generator)
    code, error = writer.attempt_policy_generation(
        initial_policy="old",
        tests="tests",
        frontier_context="frontier",
        frontier_context_details=frontier,
        optimization_brief=brief,
        failure_context_str=None,
        repair_attempt=1,
    )

    assert code is None
    assert error == "writer broke"
