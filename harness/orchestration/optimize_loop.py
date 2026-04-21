"""Explicit optimizer loop orchestration.

This module owns the top-level IC optimize control flow.  The repair state
machine remains the inner writer/pipeline retry engine.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from langfuse import LangfuseGeneration, LangfuseSpan, propagate_attributes

from harness.log import bind_logger, get_logger, short_task_label
from harness.localization.models import LCATask
from harness.pipeline.types import PipelineClassification, StepResult
from harness.prompts import WriterFeedbackFinding
from harness.telemetry.tracing import (
    flush_langfuse,
    get_langfuse_client,
    serialize_observation_metadata,
)
from harness.telemetry.evaluation_emitters import (
    emit_optimize_iteration_telemetry,
)

from .machine import (
    RepairStateMachine,
    _MAX_POLICY_REPAIRS,
    build_loop_dependencies,
)
from .types import (
    AcceptedPolicy,
    EvaluationMetrics,
    EvaluationPhaseResult,
    EvaluationResult,
    FeedbackPackage,
    IterationOutcome,
    IterationRecord,
    LoopContext,
    LoopDependencies,
    OptimizationIterationState,
    PreparedTasks,
    ReducerSummary,
    RepairContext,
    RepairMachineModel,
    RepairOutcome,
    RunMetadata,
    WriterInputContext,
)

if TYPE_CHECKING:
    from harness.telemetry.hosted.baselines import EvaluationRecord


DEFAULT_OPTIMIZATION_ITERATIONS = 5
_LOGGER = get_logger(__name__)


def _score_context_summary_from_metrics(metrics: EvaluationMetrics) -> dict[str, object]:
    metric_map = metrics.as_map()
    summary: dict[str, object] = {}
    for key in (
        "gold_hop",
        "issue_hop",
        "token_efficiency",
        "gold_min_hops",
        "issue_min_hops",
        "previous_total_token_count",
    ):
        value = metric_map.get(key)
        if isinstance(value, (int, float, bool)):
            summary[key] = value
    return summary


def _baseline_record_model(
    baseline_record: "EvaluationRecord | Mapping[str, object] | None",
) -> "EvaluationRecord | None":
    if isinstance(baseline_record, Mapping):
        from harness.telemetry.hosted.baselines import (
            EvaluationRecord as EvaluationRecordModel,
        )

        return EvaluationRecordModel.model_validate(baseline_record)
    return baseline_record


class OptimizeTaskLoop:
    """Readable per-task optimizer: evaluate -> reduce -> write -> validate."""

    def __init__(
        self,
        *,
        context: LoopContext,
        deps: LoopDependencies,
        max_policy_repairs: int = _MAX_POLICY_REPAIRS,
    ) -> None:
        self.context = context
        self.deps = deps
        self.max_policy_repairs = max_policy_repairs
        self._current_iteration_span: LangfuseSpan | LangfuseGeneration | None = None
        self._logger = bind_logger(
            _LOGGER,
            optimize_run_id=context.optimize_run_id,
            writer_session_id=context.writer_session_id,
            task=short_task_label(
                context.task,
                ordinal=context.task_ordinal,
                total=context.task_total,
            ),
            task_identity=context.task.task_id,
            repo=context.task.repo,
        )

    def _iteration_logger(
        self,
        state: OptimizationIterationState,
    ):
        return bind_logger(
            self._logger,
            iteration=state.iteration,
            iteration_ordinal=state.iteration + 1,
            iterations=state.iterations,
        )

    def run(self) -> list[IterationRecord]:
        status = "running"
        try:
            self._logger.info(
                "run_loop_started",
                iterations=self.context.iterations,
            )
            prepared_tasks = self._prepare_context()
            while self.context.current_iteration < self.context.iterations:
                outcome = self._run_iteration(prepared_tasks)
                if outcome.failed_phase:
                    status = "failed"
                    break
                if not outcome.continue_next:
                    status = "completed"
                    break
            else:
                status = "completed"
            return self.context.history
        finally:
            self._finalize_run_trace(status=status)
            self._logger.info(
                "run_loop_completed",
                status=status,
                iterations_completed=len(self.context.history),
            )

    def _prepare_context(self) -> PreparedTasks:
        self._logger.info(
            "backend_init_started",
        )
        try:
            prepared = self.deps.prepare_iteration_tasks(
                self.context.task, self.context.run_trace
            )
        except Exception:
            prepared = PreparedTasks(
                task=self.context.task,
                resolved_repo_path=self.context.task.repo,
                jc_repo_id=None,
            )
        self.context.prepared_tasks = prepared
        self._logger.info(
            "backend_init_completed",
            resolved_repo_path=prepared.resolved_repo_path,
        )
        return prepared

    def _run_iteration(self, prepared_tasks: PreparedTasks) -> IterationOutcome:
        started_at = time.perf_counter()
        state = OptimizationIterationState(
            iteration=self.context.current_iteration,
            iterations=self.context.iterations,
            task_identity=self.context.task.task_id,
            optimize_run_id=self.context.optimize_run_id,
            writer_session_id=self.context.writer_session_id,
            task_ordinal=self.context.task_ordinal,
            task_total=self.context.task_total,
        )
        metadata = self._iteration_metadata(state)
        iteration_logger = self._iteration_logger(state)
        iteration_logger.info("optimize_iteration_started")
        run_span = self.context.run_trace
        if run_span is None:
            raise RuntimeError("run_loop span missing for optimize_iteration")
        with run_span.start_as_current_observation(
            name="optimize_iteration",
            as_type="span",
            metadata=serialize_observation_metadata(metadata),
        ) as iteration_span:
            self._current_iteration_span = iteration_span
            eval_result = self._evaluate_iteration(state, prepared_tasks, iteration_span)
            reducer_summary = self._reduce_iteration(state, eval_result, iteration_span)
            if eval_result.evaluation is None or not eval_result.success:
                record = self._build_failed_evaluation_record(
                    state, eval_result, reducer_summary
                )
                self._record_iteration(record, repair_outcome=None)
                outcome = IterationOutcome(
                    state=state,
                    record=record,
                    status="evaluation_failed",
                    duration_seconds=time.perf_counter() - started_at,
                    continue_next=self._advance_iteration(),
                    failed_phase="evaluate",
                )
                self._finalize_iteration_span(iteration_span, outcome)
                return outcome

            writer_input = self._build_writer_input(
                state, eval_result.evaluation, reducer_summary, iteration_span
            )
            repair_outcome = self._writer_iteration(writer_input, iteration_span)
            record = self._build_iteration_record(
                iteration=state.iteration,
                evaluation=writer_input.evaluation,
                repair_outcome=repair_outcome,
            )
            self._record_iteration(record, repair_outcome=repair_outcome)
            if repair_outcome and repair_outcome.success:
                score_val = record.metrics.get("score")
                self.context.prev_score = (
                    float(score_val) if isinstance(score_val, (int, float)) else None
                )
                self.context.prev_classified = PipelineClassification()
            outcome = IterationOutcome(
                state=state,
                record=record,
                repair_outcome=repair_outcome,
                status="success" if repair_outcome and repair_outcome.success else "rejected",
                duration_seconds=time.perf_counter() - started_at,
                continue_next=self._advance_iteration(),
            )
            self._finalize_iteration_span(iteration_span, outcome)
            iteration_logger.info(
                "optimize_iteration_completed",
                status=outcome.status,
                pipeline_passed=record.pipeline_passed,
                continue_next=outcome.continue_next,
                rejection_reason=record.error or record.writer_error,
            )
            return outcome

    def _evaluate_iteration(
        self,
        state: OptimizationIterationState,
        prepared_tasks: PreparedTasks,
        iteration_span: LangfuseSpan | LangfuseGeneration,
    ) -> EvaluationPhaseResult:
        iteration_tasks = prepared_tasks.build_iteration_tasks()
        metadata = {
            **self._iteration_metadata(state),
            "phase": "evaluate_iteration",
            "task": iteration_tasks.task.identity.task_id(),
            "session_policy": "metadata_only",
        }
        iteration_logger = self._iteration_logger(state)
        iteration_logger.info("evaluate_iteration_started")
        evaluation: EvaluationResult | None = None
        error: str | None = None
        with iteration_span.start_as_current_observation(
            name="evaluate_iteration",
            as_type="span",
            metadata=serialize_observation_metadata(metadata),
        ) as eval_span:
            try:
                evaluation = self.deps.evaluate_policy_on_item(
                    iteration_tasks.task,
                    self.context.baseline_record,
                    eval_span,
                    state.iteration,
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            try:
                eval_span.update(
                    metadata=serialize_observation_metadata(
                        {
                            "status": "ok" if evaluation and evaluation.success else "failed",
                            "success": evaluation.success if evaluation else False,
                            "error": error,
                        }
                    )
                )
            except Exception as exc:  # noqa: BLE001
                iteration_logger.warning(
                    "langfuse_metadata_update_failed",
                    operation="evaluate_iteration.metadata",
                    observation="evaluate_iteration",
                    keys=["error", "status", "success"],
                    error=str(exc),
                )
        predicted_files_obj = (
            evaluation.ic_result.as_map().get("predicted_files") if evaluation else []
        )
        predicted_files_count = (
            len(predicted_files_obj) if isinstance(predicted_files_obj, list) else 0
        )
        iteration_logger.info(
            "evaluate_iteration_completed",
            success=evaluation.success if evaluation else False,
            score=evaluation.metrics.score if evaluation else None,
            predicted_files_count=predicted_files_count,
            error=error or (evaluation.error if evaluation else None),
        )
        return EvaluationPhaseResult(
            state=state,
            prepared_tasks=prepared_tasks,
            evaluation=evaluation,
            predicted_files_count=predicted_files_count,
            success=bool(evaluation and evaluation.success),
            error=error or (evaluation.error if evaluation else None),
        )

    def _reduce_iteration(
        self,
        state: OptimizationIterationState,
        evaluation_phase: EvaluationPhaseResult,
        iteration_span: LangfuseSpan | LangfuseGeneration,
    ) -> ReducerSummary:
        iteration_logger = self._iteration_logger(state)
        iteration_logger.info("reduce_iteration_started")
        evaluation = evaluation_phase.evaluation
        current_score = self._score_from_evaluation(evaluation)
        previous_score = self.context.prev_score
        score_delta = (
            current_score - previous_score
            if current_score is not None and previous_score is not None
            else None
        )
        regression_raw = evaluation.metrics.get("iteration_regression") if evaluation else None
        summary = ReducerSummary(
            iteration=state.iteration,
            task_identity=state.task_identity,
            current_score=current_score,
            previous_score=previous_score,
            score_delta=score_delta,
            regression=regression_raw if isinstance(regression_raw, bool) else None,
            evaluation_success=evaluation_phase.success,
            pipeline_feedback_available=self.context.prev_classified is not None,
            comparison_summary=evaluation.comparison_summary if evaluation else None,
        )
        with iteration_span.start_as_current_observation(
            name="reduce_iteration",
            as_type="span",
            metadata=serialize_observation_metadata(
                {
                    **self._iteration_metadata(state),
                    "phase": "reduce_iteration",
                    "current_score": summary.current_score,
                    "previous_score": summary.previous_score,
                    "score_delta": summary.score_delta,
                    "evaluation_success": summary.evaluation_success,
                }
            ),
        ):
            pass
        iteration_logger.info(
            "reduce_iteration_completed",
            current_score=summary.current_score,
            previous_score=summary.previous_score,
            score_delta=summary.score_delta,
        )
        return summary

    def _build_writer_input(
        self,
        state: OptimizationIterationState,
        evaluation: EvaluationResult,
        reducer_summary: ReducerSummary,
        iteration_span: LangfuseSpan | LangfuseGeneration,
    ) -> WriterInputContext:
        repo_root = self.deps.find_repo_root()
        iteration_logger = self._iteration_logger(state)
        iteration_logger.info(
            "build_writer_input_started",
            previous_score=self.context.prev_score,
        )
        with iteration_span.start_as_current_observation(
            name="build_writer_input",
            as_type="span",
            metadata=serialize_observation_metadata(
                {
                    **self._iteration_metadata(state),
                    "phase": "build_writer_input",
                    "previous_score": self.context.prev_score,
                }
            ),
        ) as feedback_span:
            feedback = self.deps.build_iteration_feedback(
                evaluation,
                self.context.prev_score,
                self.context.prev_classified,
                repo_root,
            )
            feedback = self._attach_reducer_summary(feedback, reducer_summary)
            try:
                feedback_span.update(
                    metadata=serialize_observation_metadata(
                        {
                            "tests_chars": len(feedback.tests),
                            "frontier_context_chars": len(feedback.frontier_context),
                            "reducer_summary": reducer_summary.model_dump(mode="json"),
                        }
                    )
                )
            except Exception as exc:  # noqa: BLE001
                iteration_logger.warning(
                    "langfuse_metadata_update_failed",
                    operation="build_writer_input.metadata",
                    observation="build_writer_input",
                    keys=["frontier_context_chars", "reducer_summary", "tests_chars"],
                    error=str(exc),
                )
        iteration_logger.info(
            "build_writer_input_completed",
            tests_chars=len(feedback.tests),
            frontier_context_chars=len(feedback.frontier_context),
            reducer_available=True,
        )
        return WriterInputContext(
            state=state,
            evaluation=evaluation,
            reducer_summary=reducer_summary,
            feedback=feedback,
        )

    def _writer_iteration(
        self,
        writer_input: WriterInputContext,
        iteration_span: LangfuseSpan | LangfuseGeneration,
    ) -> RepairOutcome:
        state = writer_input.state
        bind_logger(
            self._iteration_logger(state),
            writer_session_id=state.writer_session_id,
        ).info(
            "writer_iteration_started",
        )
        with iteration_span.start_as_current_observation(
            name="writer_iteration",
            as_type="span",
            metadata=serialize_observation_metadata(
                {
                    **self._iteration_metadata(state),
                    "phase": "writer_iteration",
                    "writer_session_id": state.writer_session_id,
                }
            ),
        ) as writer_span:
            repair_outcome = self._run_repair_phase(
                self.deps.find_repo_root(),
                writer_input.evaluation,
                writer_input.feedback,
                writer_span,
            )
            try:
                writer_span.update(
                    metadata=serialize_observation_metadata(
                        {
                            "pipeline_passed": repair_outcome.success,
                            "attempts_used": repair_outcome.attempts_used,
                            "writer_error": repair_outcome.writer_error,
                            "error": repair_outcome.error,
                        }
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._iteration_logger(state).warning(
                    "langfuse_metadata_update_failed",
                    operation="writer_iteration.metadata",
                    observation="writer_iteration",
                    keys=["attempts_used", "error", "pipeline_passed", "writer_error"],
                    error=str(exc),
                )
        self._iteration_logger(state).info(
            "writer_iteration_completed",
            pipeline_passed=repair_outcome.success,
            attempts_used=repair_outcome.attempts_used,
        )
        return repair_outcome

    def _run_repair_phase(
        self,
        repo_root: Path,
        evaluation: EvaluationResult,
        feedback: FeedbackPackage,
        iteration_span: LangfuseSpan | LangfuseGeneration | None,
    ) -> RepairOutcome:
        repair_ctx = RepairContext(
            repo_root=repo_root,
            evaluation=evaluation,
            feedback=feedback,
            max_repair_attempts=self.max_policy_repairs,
            parent_trace=iteration_span,
            writer_model=self.deps.get_writer_model(),
            optimize_run_id=self.context.optimize_run_id,
            writer_session_id=self.context.writer_session_id,
            task_identity=self.context.task.task_id,
            iteration=self.context.current_iteration,
            iterations=self.context.iterations,
            task_ordinal=self.context.task_ordinal,
            task_total=self.context.task_total,
            pipeline=self.deps.default_pipeline(),
        )
        repair_model = RepairMachineModel(
            context=repair_ctx,
            deps=self.deps,
        )
        return RepairStateMachine(repair_model).run()

    def _build_iteration_record(
        self,
        iteration: int,
        evaluation: EvaluationResult,
        repair_outcome: RepairOutcome | None,
    ) -> IterationRecord:
        pipeline_feedback = (
            repair_outcome.pipeline_feedback if repair_outcome else self.context.last_classified
        )
        repair_attempts_val = (
            repair_outcome.repair_attempts_reported if repair_outcome else None
        )
        max_repairs_val = (
            repair_outcome.max_policy_repairs if repair_outcome else self.max_policy_repairs
        )
        error_val = repair_outcome.error if repair_outcome else None
        writer_err_val = repair_outcome.writer_error if repair_outcome else None
        record = IterationRecord(
            iteration=iteration,
            metrics=evaluation.metrics,
            score_components=evaluation.score_components,
            pipeline_passed=repair_outcome.success if repair_outcome else None,
            pipeline_feedback=pipeline_feedback,
            repair_attempts=repair_attempts_val
            if isinstance(repair_attempts_val, int)
            else (repair_outcome.attempts_used if repair_outcome else None),
            max_policy_repairs=max_repairs_val
            if isinstance(max_repairs_val, int)
            else self.max_policy_repairs,
            error=error_val if isinstance(error_val, str) else None,
            writer_error=writer_err_val if isinstance(writer_err_val, str) else None,
            failed_step=repair_outcome.failed_step if repair_outcome else None,
            failed_exit_code=repair_outcome.failed_exit_code if repair_outcome else None,
            failed_summary=repair_outcome.failed_summary if repair_outcome else None,
        )
        failed_step = (
            self._first_failed_step(repair_outcome.pipeline_results)
            if repair_outcome
            else None
        )
        if failed_step:
            record.failed_step = failed_step.name
            record.failed_exit_code = failed_step.exit_code
            record.failed_summary = failed_step.stderr or failed_step.stdout
        if repair_outcome and repair_outcome.success:
            record.accepted_policy = AcceptedPolicy(
                code=repair_outcome.policy,
                source=repair_outcome.accepted_policy_source,
                changed_by_pipeline=repair_outcome.accepted_policy_changed_by_pipeline,
                repair_attempts=record.repair_attempts or repair_outcome.attempts_used,
                max_policy_repairs=record.max_policy_repairs or self.max_policy_repairs,
            )
        return record

    def _build_failed_evaluation_record(
        self,
        state: OptimizationIterationState,
        evaluation_phase: EvaluationPhaseResult,
        reducer_summary: ReducerSummary,
    ) -> IterationRecord:
        del reducer_summary
        evaluation = evaluation_phase.evaluation
        metrics = evaluation.metrics if evaluation else EvaluationMetrics()
        return IterationRecord(
            iteration=state.iteration,
            metrics=metrics,
            score_components=evaluation.score_components if evaluation else {},
            pipeline_passed=None,
            error=evaluation_phase.error or "evaluation_failed",
        )

    def _record_iteration(
        self,
        record: IterationRecord,
        repair_outcome: RepairOutcome | None,
    ) -> None:
        outcome = repair_outcome
        pipeline_val = record.pipeline_passed
        if outcome and pipeline_val is False:
            if outcome.pipeline_results:
                classified = outcome.pipeline_feedback or self.deps.classify_results(
                    outcome.pipeline_results
                )
                record.pipeline_feedback = record.pipeline_feedback or classified
                failed_step = self._first_failed_step(outcome.pipeline_results)
                if failed_step:
                    record.failed_step = record.failed_step or failed_step.name
                    record.failed_exit_code = record.failed_exit_code or failed_step.exit_code
                    record.failed_summary = (
                        record.failed_summary or failed_step.stderr or failed_step.stdout
                    )
        if pipeline_val is False:
            if outcome and outcome.pipeline_results:
                classified = outcome.pipeline_feedback or self.deps.classify_results(
                    outcome.pipeline_results
                )
                record.pipeline_feedback = classified
            if record.failed_step is None:
                if outcome and outcome.pipeline_results:
                    failed_step = self._first_failed_step(outcome.pipeline_results)
                    if failed_step:
                        record.failed_step = failed_step.name
                        record.failed_exit_code = failed_step.exit_code
                        record.failed_summary = failed_step.stderr or failed_step.stdout
                if record.failed_step is None:
                    record.failed_step = "pipeline_failed"
        if record.pipeline_feedback is None and outcome and outcome.pipeline_results:
            record.pipeline_feedback = self.deps.classify_results(outcome.pipeline_results)
        if record.writer_error is None and outcome:
            if isinstance(outcome.writer_error, str):
                record.writer_error = outcome.writer_error
            elif outcome.success is False:
                record.writer_error = "writer_failed"
        self.context.history.append(record)
        self._record_iteration_scores(self._current_iteration_span, record)

    def _record_iteration_scores(
        self,
        handle: LangfuseSpan | LangfuseGeneration | None,
        record: IterationRecord,
    ) -> None:
        emit_optimize_iteration_telemetry(
            handle,
            metrics=record.metrics,
            score_component_states=record.score_components,
            iteration=record.iteration,
            status="recorded",
            pipeline_passed=record.pipeline_passed,
            score_context_summary=_score_context_summary_from_metrics(record.metrics),
            error=record.error or record.writer_error,
        )

    def _advance_iteration(self) -> bool:
        self.context.current_iteration += 1
        return self.context.current_iteration < self.context.iterations

    def _finalize_iteration_span(
        self,
        span: LangfuseSpan | LangfuseGeneration,
        outcome: IterationOutcome,
    ) -> None:
        metadata: dict[str, object] = {
            "status": outcome.status,
            "iteration": outcome.state.iteration,
            "continue_next": outcome.continue_next,
        }
        if outcome.record.pipeline_passed is not None:
            metadata["pipeline_passed"] = outcome.record.pipeline_passed
        if outcome.failed_phase:
            metadata["failed_phase"] = outcome.failed_phase
        failure_val = outcome.record.error or outcome.record.writer_error
        if failure_val:
            metadata["error"] = failure_val
        if outcome.duration_seconds is not None:
            metadata["duration_seconds"] = outcome.duration_seconds
        try:
            span.update(metadata=serialize_observation_metadata(metadata))
        except Exception as exc:  # noqa: BLE001
            self._iteration_logger(outcome.state).warning(
                "langfuse_metadata_update_failed",
                operation="optimize_iteration.finalize",
                observation="optimize_iteration",
                keys=sorted(metadata.keys()),
                error=str(exc),
            )
        self._current_iteration_span = None

    def _finalize_run_trace(self, status: str | None) -> None:
        span = self.context.run_trace
        if span is None:
            return
        metadata: dict[str, object] = {
            "iterations_completed": len(self.context.history),
        }
        if status:
            metadata["status"] = status
        try:
            span.update(metadata=serialize_observation_metadata(metadata))
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "langfuse_metadata_update_failed",
                operation="run_loop.finalize",
                observation="run_loop",
                keys=sorted(metadata.keys()),
                error=str(exc),
            )

    def _iteration_metadata(self, state: OptimizationIterationState) -> dict[str, object]:
        metadata: dict[str, object] = {
            "iteration": state.iteration,
            "iteration_ordinal": state.iteration + 1,
            "iterations": state.iterations,
            "task_identity": state.task_identity,
        }
        if state.optimize_run_id:
            metadata["optimize_run_id"] = state.optimize_run_id
        if state.writer_session_id:
            metadata["writer_session_id"] = state.writer_session_id
        if state.task_ordinal is not None:
            metadata["ordinal"] = state.task_ordinal
        if state.task_total is not None:
            metadata["total"] = state.task_total
        return metadata

    def _attach_reducer_summary(
        self,
        feedback: FeedbackPackage,
        reducer_summary: ReducerSummary,
    ) -> FeedbackPackage:
        finding = WriterFeedbackFinding(
            kind="global_reducer_summary",
            message=reducer_summary.compact_message(),
            severity="info",
            source="optimizer.reducer",
        )
        brief = feedback.optimization_brief.model_copy(
            update={
                "findings": [
                    *feedback.optimization_brief.findings,
                    finding,
                ]
            }
        )
        return feedback.model_copy(update={"optimization_brief": brief})

    def _score_from_evaluation(self, evaluation: EvaluationResult | None) -> float | None:
        if evaluation is None:
            return None
        score = evaluation.metrics.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return None

    def _first_failed_step(self, steps: Sequence[StepResult]) -> StepResult | None:
        for step in steps:
            if step.exit_code != 0:
                return step
        return None


def run_loop(
    task: LCATask,
    iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
    parent_trace: LangfuseSpan | LangfuseGeneration | None = None,
    baseline_record: "EvaluationRecord | Mapping[str, object] | None" = None,
    session_id: str | None = None,
    optimize_run_id: str | None = None,
    task_ordinal: int | None = None,
    task_total: int | None = None,
    flush_tracing: bool = True,
) -> list[IterationRecord]:
    """Closed-loop optimizer: evaluate, reduce, write, validate, repeat."""
    writer_session_id = session_id
    root_meta = RunMetadata(
        task=task,
        iterations=iterations,
        optimize_run_id=optimize_run_id,
        writer_session_id=writer_session_id,
        task_ordinal=task_ordinal,
        task_total=task_total,
    )
    context: LoopContext | None = None
    deps: LoopDependencies | None = None
    history: list[IterationRecord] = []
    if parent_trace is None:
        run_loop_cm = get_langfuse_client().start_as_current_observation(
            name="run_loop",
            as_type="span",
            input=task.model_dump(),
            metadata=serialize_observation_metadata(root_meta.model_dump()),
        )
    else:
        run_loop_cm = parent_trace.start_as_current_observation(
            name="run_loop",
            as_type="span",
            input=task.model_dump(),
            metadata=serialize_observation_metadata(root_meta.model_dump()),
        )
    with run_loop_cm as run_span:
        with propagate_attributes(
            trace_name="run_loop",
            session_id=writer_session_id,
        ):
            context = LoopContext(
                task=task,
                iterations=iterations,
                session_id=writer_session_id,
                optimize_run_id=optimize_run_id,
                writer_session_id=writer_session_id,
                task_ordinal=task_ordinal,
                task_total=task_total,
                parent_trace=parent_trace,
                run_metadata=root_meta,
                baseline_record=_baseline_record_model(baseline_record),
                run_trace=run_span,
            )
            deps = build_loop_dependencies()
            try:
                history = OptimizeTaskLoop(
                    context=context,
                    deps=deps,
                    max_policy_repairs=_MAX_POLICY_REPAIRS,
                ).run()
            finally:
                if flush_tracing:
                    flush_langfuse()
    if not history and context is not None and deps is not None:
        history = _fallback_history_from_context(context, deps, parent_trace)
    return history


def _fallback_history_from_context(
    context: LoopContext,
    deps: LoopDependencies,
    parent_trace: LangfuseSpan | LangfuseGeneration | None,
) -> list[IterationRecord]:
    repo_root = deps.find_repo_root()
    pipeline = deps.default_pipeline()
    fallback_parent = context.run_trace or parent_trace
    pipeline_results, pipeline_success = deps.run_policy_pipeline(
        pipeline,
        repo_root,
        repair_obs=fallback_parent,
        allow_no_parent=fallback_parent is None,
    )
    classified = deps.classify_results(pipeline_results)
    failed_step = next((step for step in pipeline_results if step.exit_code != 0), None)
    return [
        IterationRecord(
            iteration=0,
            metrics=EvaluationMetrics.model_validate({}),
            pipeline_passed=pipeline_success,
            pipeline_feedback=classified,
            failed_step=failed_step.name if failed_step else "pipeline_failed",
            failed_exit_code=failed_step.exit_code if failed_step else None,
            failed_summary=(failed_step.stderr or failed_step.stdout)
            if failed_step
            else None,
            repair_attempts=_MAX_POLICY_REPAIRS,
            max_policy_repairs=_MAX_POLICY_REPAIRS,
            writer_error="writer_failed" if not pipeline_success else None,
        )
    ]
