from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langfuse import LangfuseGeneration, LangfuseSpan

from harness.log import bind_logger, get_logger, short_task_label
from harness.localization.models import LCATask
from harness.localization.runtime.evaluate import (
    evaluate_localization_batch,
    prediction_filenames,
)
from harness.orchestration.types import (
    EvaluationMetrics,
    EvaluationResult,
    ICResult,
    JCResult,
)
from harness.orchestration.runtime_context import optimization_policy_path
from harness.prompts import WriterScoreSummary, build_writer_score_summary_from_bundle
from harness.telemetry.observability_models import score_bundle_component_states

if TYPE_CHECKING:
    from harness.telemetry.hosted.baselines import EvaluationRecord


_POLICY_PATH = Path(__file__).resolve().parent.parent / "policy" / "current.py"
_LOGGER = get_logger(__name__)


def _read_policy(path: Path = _POLICY_PATH) -> str:
    path = optimization_policy_path(path)
    return path.read_text(encoding="utf-8")


def _failure_evaluation_result(error: str) -> EvaluationResult:
    failure_metrics: dict[str, float | int | bool | None] = {
        "score": -10.0,
        "iteration_regression": False,
    }
    return EvaluationResult(
        metrics=EvaluationMetrics.model_validate(failure_metrics),
        ic_result=ICResult(entries=[]),
        jc_result=JCResult(entries=[]),
        jc_metrics=EvaluationMetrics(),
        comparison_summary=None,
        policy_code="",
        success=False,
        error=error,
        score_summary=WriterScoreSummary(
            primary_name="composed_score",
            primary_score=-10.0,
            composed_score=-10.0,
            optimization_goal="maximize",
        ),
        score_components={},
    )


def evaluate_policy_on_item(
    task: LCATask,
    baseline_record: "EvaluationRecord | None",
    iteration_span: LangfuseSpan | LangfuseGeneration | None = None,
    iteration_index: int | None = None,
) -> EvaluationResult:
    del baseline_record
    logger = bind_logger(
        _LOGGER,
        task=short_task_label(task),
        task_identity=task.task_id,
        iteration=iteration_index,
        repo=task.repo,
    )
    logger.info("policy_evaluation_started")
    eval_result = evaluate_localization_batch(
        tasks=[task],
        dataset_provenance=None,
        worktree_manager=None,
        parent_trace=iteration_span,
    )
    if eval_result.failure:
        category = getattr(
            eval_result.failure.category,
            "value",
            eval_result.failure.category,
        )
        logger.warning(
            "policy_evaluation_failed",
            failure_category=category,
            failure_message=eval_result.failure.message,
        )
        return _failure_evaluation_result(f"{category}: {eval_result.failure.message}")
    if not eval_result.items:
        logger.warning(
            "policy_evaluation_failed",
            failure_category="evaluation_failed",
            failure_message="no_results",
        )
        return _failure_evaluation_result("evaluation_failed: no_results")

    task_result = eval_result.items[0]
    score_bundle = task_result.score_bundle
    score_value = (
        score_bundle.composed_score
        if score_bundle.composed_score is not None
        else -10.0
    )
    additional_metrics = [
        {"name": name, "value": result.value}
        for name, result in score_bundle.results.items()
        if result.value is not None
    ]
    eval_metrics = EvaluationMetrics.model_validate(
        {
            "score": score_value,
            "iteration_regression": False,
            "additional_metrics": additional_metrics,
        }
    )
    logger.info(
        "policy_evaluation_completed",
        score=score_value,
        control_score=eval_result.machine_score,
        predicted_files_count=len(prediction_filenames(task_result.prediction)),
        component_metric_names=sorted(entry["name"] for entry in additional_metrics),
    )
    return EvaluationResult(
        metrics=eval_metrics,
        ic_result=ICResult.model_validate(
            {"predicted_files": prediction_filenames(task_result.prediction)}
        ),
        jc_result=JCResult(entries=[]),
        jc_metrics=EvaluationMetrics(),
        comparison_summary=None,
        policy_code=_read_policy(),
        control_score=eval_result.machine_score,
        score_summary=build_writer_score_summary_from_bundle(score_bundle),
        score_components=score_bundle_component_states(score_bundle),
    )
