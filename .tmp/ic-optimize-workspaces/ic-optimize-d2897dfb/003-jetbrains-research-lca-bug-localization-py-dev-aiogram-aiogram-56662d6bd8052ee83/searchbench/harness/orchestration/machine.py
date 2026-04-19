"""
State machines and lifecycle for optimization/repair loops.
Holds state transitions and callbacks; no CLI or entrypoint concerns.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

from statemachine import State, StateChart
from statemachine.event import BoundEvent

import jcodemunch_mcp.tools.index_folder as index_folder_tool

from harness import pipeline as pipeline_module
from harness.agents import writer as agent_writer
from harness.localization.models import LCATask
from harness.pipeline.types import PipelineClassification, StepResult
from harness.prompts import WriterGuidance
from harness.telemetry import tracing
from harness.telemetry.tracing import score_emitter
from harness.utils import diff as diff_utils
from harness.utils import env as env_utils
from harness.utils import repo_root as repo_root_utils
from harness.utils import test_loader
from harness.utils import type_loader
from . import dependencies as dependency_wiring
from . import evaluation as policy_evaluation
from . import feedback as feedback_wiring
from . import writer as policy_writer
from .runtime_context import optimization_repo_root
from .listeners import OptimizationTracingListener, RepairTracingListener
from .types import (
    AcceptedPolicy,
    EvaluationMetrics,
    EvaluationResult,
    FeedbackPackage,
    IterationRecord,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
    RepairOutcome,
    RunMetadata,
    SpanHandle,
)

if TYPE_CHECKING:
    from harness.telemetry.hosted.baselines import EvaluationRecord

_BOUND_EVENT = cast(Any, BoundEvent)
_MAX_POLICY_REPAIRS = 3
_LOGGER = logging.getLogger(__name__)

# The installed python-statemachine release still reads ``BoundEvent.key``
# internally while dispatching callbacks. Keep the patch isolated here; harness
# callbacks and listeners use state hooks and EventData instead.
if not hasattr(BoundEvent, "key"):  # pragma: no cover - depends on library release
    setattr(_BOUND_EVENT, "key", property(lambda self: self.name))  # type: ignore[attr-defined]


class RepairStateMachine(StateChart[RepairMachineModel]):
    """
    Drives the candidate generation + pipeline repair loop using state-driven callbacks.

    Construction is cold — the machine starts in ``idle``.  Calling ``run()``
    fires the ``start`` event, which transitions to ``generating_candidate``
    through the normal state-entry path so listeners own span lifecycle for
    every attempt including the first.
    """

    catch_errors_as_events = True

    idle = State(initial=True)
    generating_candidate = State()
    running_pipeline = State()
    retrying = State()
    candidate_valid = State(final=True)
    repair_exhausted = State(final=True)

    start = idle.to(generating_candidate)
    writer_generated = generating_candidate.to(running_pipeline)
    writer_retryable_failure = generating_candidate.to(retrying)
    writer_exhausted = generating_candidate.to(repair_exhausted)
    pipeline_passed = running_pipeline.to(candidate_valid)
    pipeline_retryable_failure = running_pipeline.to(retrying)
    pipeline_exhausted = running_pipeline.to(repair_exhausted)
    retry = retrying.to(generating_candidate)

    @property
    def context(self) -> RepairContext:
        return self.model.context

    @property
    def deps(self) -> LoopDependencies:
        return self.model.deps

    def __init__(self, model: RepairMachineModel):
        super().__init__(
            model=model,
            listeners=[
                RepairTracingListener(),
            ],
        )

    # Entry point ---------------------------------------------------------
    def run(self) -> RepairOutcome:
        self.raise_("start")  # type: ignore[call-arg]
        attempts_used = self.context.attempts_used
        final_attempts = (
            self.context.max_repair_attempts
            if self.context.pipeline_passed is False
            else min(attempts_used, self.context.max_repair_attempts)
        )
        self.context.attempts_used = final_attempts
        self.context.repair_attempts_reported = (
            max(0, final_attempts - 1)
            if self.context.pipeline_passed
            else self.context.max_repair_attempts
        )
        self.context.max_policy_repairs_reported = self.context.max_repair_attempts
        return RepairOutcome(
            policy=self.deps.read_policy(),
            pipeline_results=self.context.pipeline_results,
            success=self.context.pipeline_passed,
            attempts_used=final_attempts,
            pipeline_feedback=self.context.pipeline_feedback or self.context.last_classified,
            writer_error=self.context.writer_error,
            error=self.context.error,
            failed_step=self.context.failed_step,
            failed_exit_code=self.context.failed_exit_code,
            failed_summary=self.context.failed_summary,
            accepted_policy_source=self.context.accepted_policy_source,
            accepted_policy_changed_by_pipeline=self.context.accepted_policy_changed_by_pipeline,
            repair_attempts_reported=self.context.repair_attempts_reported,
            max_policy_repairs=self.context.max_policy_repairs_reported,
        )

    # Writer/pipeline execution ------------------------------------------
    def on_enter_generating_candidate(self, event: object, state: State) -> None:
        self._run_writer()

    def on_enter_retrying(self, event: object, state: State) -> None:
        if self._attempts_remaining():
            self.raise_("retry")  # type: ignore[call-arg]
        else:
            self.raise_("writer_exhausted")  # type: ignore[call-arg]

    def _attempts_remaining(self) -> bool:
        return self.context.attempts_used < self.context.max_repair_attempts

    def _run_writer(self) -> None:
        ctx = self.context
        deps = self.deps
        guidance = list(ctx.feedback.optimization_brief.guidance.instructions)
        mode = ctx.feedback.optimization_brief.guidance.mode
        if ctx.attempts_used > 0:
            mode = "repair"
            guidance.append(
                "Repair mode: produce the smallest change that makes harness/policy.py pass ruff, basedpyright, and pytest."
            )
            guidance.append("Add missing parameter annotations and return type -> float before optimizing frontier_priority.")
        if ctx.last_classified and ctx.last_classified.type_errors:
            guidance.append("Fix all type errors and ensure parameters are annotated; avoid unknown types.")
        optimization_brief = ctx.feedback.optimization_brief.model_copy(
            update={"guidance": WriterGuidance(mode=mode, instructions=guidance)}
        )

        _LOGGER.info(
            "[OPTIMIZE] writer_generation start attempt=%s model=%s failure_context_chars=%s",
            ctx.attempts_used,
            ctx.writer_model or "unknown",
            len(ctx.failure_context or ""),
        )
        with deps.start_observation(
            name="writer_generation",
            parent=ctx.repair_observation,
            metadata={
                "attempt": ctx.attempts_used,
                "writer_model": ctx.writer_model,
                "failure_context_chars": len(ctx.failure_context or ""),
            },
        ) as writer_span:
            new_code, writer_error = deps.attempt_policy_generation(
                initial_policy=ctx.evaluation.policy_code,
                tests=ctx.feedback.tests,
                frontier_context=ctx.feedback.frontier_context,
                frontier_context_details=ctx.feedback.frontier_context_details,
                optimization_brief=optimization_brief,
                failure_context_str=ctx.failure_context,
                repair_attempt=ctx.attempts_used,
                parent_trace=writer_span,
                pipeline_feedback=ctx.last_classified,
            )
            try:
                writer_span.update(
                    metadata={
                        "attempt": ctx.attempts_used,
                        "code_returned": new_code is not None,
                        "writer_error": writer_error,
                    }
                )
            except Exception:
                pass

        if new_code is None:
            _LOGGER.info(
                "[OPTIMIZE] writer_generation complete attempt=%s code_returned=false error=%s",
                ctx.attempts_used,
                writer_error,
            )
            ctx.writer_error = writer_error
            ctx.error = writer_error or "writer_failed"
            ctx.attempts_used += 1
            ctx.failure_context = self._format_failure(
                [], writer_error, ctx.attempts_used
            )
            if self._attempts_remaining():
                self.raise_("writer_retryable_failure")  # type: ignore[call-arg]
            else:
                self.raise_("writer_exhausted")  # type: ignore[call-arg]
            return

        deps.write_policy(new_code)
        ctx.candidate_code = new_code
        _LOGGER.info(
            "[OPTIMIZE] writer_generation complete attempt=%s code_returned=true code_chars=%s",
            ctx.attempts_used,
            len(new_code),
        )
        self.raise_("writer_generated")  # type: ignore[call-arg]
        self._run_pipeline()

    def _run_pipeline(self) -> None:
        ctx = self.context
        deps = self.deps
        pipeline = ctx.pipeline or deps.default_pipeline()
        check_names = [getattr(step, "name", "unknown") for step in getattr(pipeline, "steps", [])]
        _LOGGER.info(
            "[OPTIMIZE] pipeline_check start attempt=%s checks=%s repo_root=%s",
            ctx.attempts_used,
            ",".join(check_names) if check_names else "unknown",
            ctx.repo_root,
        )
        with deps.start_observation(
            name="pipeline_check",
            parent=ctx.repair_observation,
            metadata={
                "attempt": ctx.attempts_used,
                "checks": check_names,
                "repo_root": str(ctx.repo_root),
            },
        ) as pipeline_span:
            pipeline_results, pipeline_success = deps.run_policy_pipeline(
                pipeline,
                ctx.repo_root,
                repair_obs=pipeline_span,
            )
            try:
                pipeline_span.update(
                    metadata={
                        "pipeline_passed": pipeline_success,
                        "checks_run": [step.name for step in pipeline_results],
                    }
                )
            except Exception:
                pass
        # Enforce canonical step result shape at the boundary.
        ctx.pipeline_results = [step if isinstance(step, StepResult) else StepResult.from_external(step) for step in pipeline_results]
        ctx.pipeline_passed = pipeline_success
        ctx.attempts_used += 1
        _LOGGER.info(
            "[OPTIMIZE] pipeline_check result attempt=%s passed=%s checks_run=%s",
            ctx.attempts_used - 1,
            pipeline_success,
            ",".join(step.name for step in ctx.pipeline_results),
        )

        if pipeline_success:
            with deps.start_observation(
                name="policy_acceptance",
                parent=ctx.repair_observation,
                metadata={"attempts_used": ctx.attempts_used, "status": "accepted"},
            ) as acceptance_span:
                final_policy, success_meta = deps.finalize_successful_policy(
                    new_code=ctx.candidate_code or deps.read_policy(),
                    attempts_used=ctx.attempts_used,
                    max_repair_attempts=ctx.max_repair_attempts,
                    repair_obs=acceptance_span,
                )
                try:
                    acceptance_span.update(
                        metadata=success_meta.model_dump(mode="json")
                    )
                except Exception:
                    pass
            ctx.candidate_code = final_policy
            ctx.accepted_policy_source = success_meta.accepted_policy_source
            ctx.accepted_policy_changed_by_pipeline = success_meta.accepted_policy_changed_by_pipeline
            ctx.repair_attempts_reported = success_meta.repair_attempts
            ctx.max_policy_repairs_reported = success_meta.max_policy_repairs
            self.raise_("pipeline_passed")  # type: ignore[call-arg]
            return

        classified = deps.classify_results(pipeline_results)
        ctx.last_classified = classified
        ctx.pipeline_feedback = classified
        failure_details = deps.finalize_failed_repair(
            pipeline_results=pipeline_results,
            classified=classified,
            writer_model=ctx.writer_model,
            attempts_used=ctx.attempts_used,
            writer_error=ctx.writer_error,
        )
        ctx.failure_context = failure_details.failure_context
        ctx.evaluation.policy_code = failure_details.policy_code
        ctx.failed_step = failure_details.failed_step
        ctx.failed_exit_code = failure_details.failed_exit_code
        ctx.failed_summary = failure_details.failed_summary
        ctx.error = failure_details.error
        if self._attempts_remaining():
            _LOGGER.info(
                "[OPTIMIZE] pipeline_check rejected attempt=%s retrying=true failed_step=%s",
                ctx.attempts_used - 1,
                ctx.failed_step or "unknown",
            )
            self.raise_("pipeline_retryable_failure")  # type: ignore[call-arg]
        else:
            _LOGGER.info(
                "[OPTIMIZE] pipeline_check rejected attempt=%s retrying=false failed_step=%s",
                ctx.attempts_used - 1,
                ctx.failed_step or "unknown",
            )
            self.raise_("pipeline_exhausted")  # type: ignore[call-arg]

    # Helpers -------------------------------------------------------------
    def _format_failure(self, pipeline_results: list[StepResult], writer_error: str | None, attempt: int) -> str:
        return self.deps.format_pipeline_failure_context(
            pipeline_results,
            self.context.last_classified,
            writer_error,
            self.context.writer_model,
            attempt,
        )


class OptimizationStateMachine(StateChart[OptimizationMachineModel]):
    """Top-level orchestrator for a full loop run driven by state callbacks."""

    preparing = State(initial=True)
    evaluating = State()
    repairing = State()
    accepted_round = State()
    completed = State(final=True)
    failed = State(final=True)

    prepare = preparing.to(evaluating, cond="prep_ready") | preparing.to(failed)
    evaluation_done = evaluating.to(
        repairing, cond="evaluation_successful"
    ) | evaluating.to(accepted_round, unless="evaluation_successful")
    repair_done = repairing.to(accepted_round)
    advance = accepted_round.to(
        evaluating, cond="has_more_iterations"
    ) | accepted_round.to(completed)

    @property
    def context(self) -> LoopContext:
        return self.model.context

    @property
    def deps(self) -> LoopDependencies:
        return self.model.deps

    @property
    def max_policy_repairs(self) -> int:
        return self.model.max_policy_repairs

    def __init__(self, model: OptimizationMachineModel):
        self._ensure_run_trace(model)
        super().__init__(
            model=model,
            listeners=[
                OptimizationTracingListener(),
            ],
        )

    # Entry point ---------------------------------------------------------
    def run(self) -> list[IterationRecord]:
        status: str | None = None
        try:
            _LOGGER.info(
                "[OPTIMIZE] run_loop start task=%s iterations=%s repo=%s",
                self.context.task.task_id,
                self.context.iterations,
                self.context.task.repo,
            )
            self._prepare_context()
            self.raise_("prepare")  # type: ignore[call-arg]
            status = getattr(self.current_state, "id", None)
            return self.context.history
        finally:
            self._finalize_run_trace(status=status)
            _LOGGER.info(
                "[OPTIMIZE] run_loop complete task=%s status=%s iterations_completed=%s",
                self.context.task.task_id,
                status or "unknown",
                len(self.context.history),
            )

    # Guards --------------------------------------------------------------
    def prep_ready(self) -> bool:
        return self.model.prep_ok

    def evaluation_successful(self) -> bool:
        return bool(self.model.current_evaluation and self.model.current_evaluation.success)

    def has_more_iterations(self) -> bool:
        return self.context.current_iteration + 1 < self.context.iterations

    # Run root ------------------------------------------------------------
    def _ensure_run_trace(self, model: OptimizationMachineModel) -> None:
        ctx = model.context
        if ctx.run_trace is None:
            raise RuntimeError("run_trace is required before starting the optimization state machine")

    def _finalize_run_trace(self, status: str | None) -> None:
        span = self.context.run_trace
        if span is None:
            return
        metadata: dict[str, object] = {"iterations_completed": len(self.context.history)}
        if status:
            metadata["status"] = status
        updater = getattr(span, "update", None)
        if callable(updater):
            try:
                updater(metadata=metadata)
            except Exception:
                pass

    # Preparation ---------------------------------------------------------
    def _prepare_context(self) -> None:
        try:
            _LOGGER.info(
                "[OPTIMIZE] backend_init start task=%s repo=%s",
                self.context.task.task_id,
                self.context.task.repo,
            )
            self.context.prepared_tasks = self.deps.prepare_iteration_tasks(
                self.context.task, self.context.run_trace
            )
            self.model.prep_ok = self.context.prepared_tasks is not None
        except Exception:
            self.context.prepared_tasks = None
            self.model.prep_ok = False
        if self.context.prepared_tasks is None:
            self.context.prepared_tasks = PreparedTasks(
                task=self.context.task,
                resolved_repo_path=self.context.task.repo,
                jc_repo_id=None,
            )
            self.model.prep_ok = True
        _LOGGER.info(
            "[OPTIMIZE] backend_init complete task=%s resolved_repo=%s",
            self.context.task.task_id,
            self.context.prepared_tasks.resolved_repo_path if self.context.prepared_tasks else None,
        )

    # Evaluation ----------------------------------------------------------
    def on_enter_evaluating(self, event: object, state: State) -> None:
        self.model.current_evaluation = None
        self.model.current_repair_outcome = None
        prepared_tasks = getattr(self.context, "prepared_tasks", None)
        if prepared_tasks is None:
            self.model.current_evaluation = None
            self.raise_("evaluation_done")  # type: ignore[call-arg]
            return

        iteration_tasks = prepared_tasks.build_iteration_tasks()
        eval_metadata = {
            "iteration": self.context.current_iteration,
            "task": iteration_tasks.task.identity.task_id(),
        }
        _LOGGER.info(
            "[OPTIMIZE] evaluate_policy start iteration=%s task=%s",
            self.context.current_iteration,
            iteration_tasks.task.identity.task_id(),
        )
        with self.deps.start_observation(
            name="evaluate_policy",
            parent=self.model.current_iteration_span,
            metadata=eval_metadata,
        ) as eval_span:
            self.model.current_evaluation = self.deps.evaluate_policy_on_item(
                iteration_tasks.task,
                self.context.baseline_record,
                eval_span,
                self.context.current_iteration,
            )
            if hasattr(eval_span, "update"):
                try:
                    eval_span.update(
                        metadata={
                            "status": "ok",
                            "success": self.model.current_evaluation.success
                            if self.model.current_evaluation
                            else False,
                        }
                    )  # type: ignore[call-arg]
                except Exception:
                    pass
        predicted_files_obj = (
            self.model.current_evaluation.ic_result.as_map().get("predicted_files")
            if self.model.current_evaluation
            else []
        )
        predicted_files_count = (
            len(predicted_files_obj) if isinstance(predicted_files_obj, list) else 0
        )
        _LOGGER.info(
            "[OPTIMIZE] evaluate_policy complete iteration=%s success=%s score=%s predicted_files=%s",
            self.context.current_iteration,
            self.model.current_evaluation.success if self.model.current_evaluation else False,
            self.model.current_evaluation.metrics.score if self.model.current_evaluation else None,
            predicted_files_count,
        )
        self.raise_("evaluation_done")  # type: ignore[call-arg]

    # Repair --------------------------------------------------------------
    def on_enter_repairing(self, event: object, state: State) -> None:
        evaluation = self.model.current_evaluation
        if evaluation is None:
            self.model.current_repair_outcome = None
            self.raise_("repair_done")  # type: ignore[call-arg]
            return

        repo_root = self.deps.find_repo_root()
        _LOGGER.info(
            "[OPTIMIZE] build_feedback start iteration=%s prev_score=%s",
            self.context.current_iteration,
            self.context.prev_score,
        )
        with self.deps.start_observation(
            name="build_feedback",
            parent=self.model.current_iteration_span,
            metadata={
                "iteration": self.context.current_iteration,
                "prev_score": self.context.prev_score,
            },
        ) as feedback_span:
            feedback = self.deps.build_iteration_feedback(
                evaluation,
                self.context.prev_score,
                self.context.prev_classified,
                repo_root,
            )
            try:
                feedback_span.update(
                    metadata={
                        "tests_chars": len(feedback.tests),
                        "frontier_context_chars": len(feedback.frontier_context),
                    }
                )
            except Exception:
                pass
        _LOGGER.info(
            "[OPTIMIZE] build_feedback complete iteration=%s tests_chars=%s frontier_context_chars=%s",
            self.context.current_iteration,
            len(feedback.tests),
            len(feedback.frontier_context),
        )
        repair_outcome = self._run_repair_phase(
            repo_root, evaluation, feedback, self.model.current_iteration_span
        )
        self.model.current_repair_outcome = repair_outcome
        self.raise_("repair_done")  # type: ignore[call-arg]

    # Acceptance / recording ---------------------------------------------
    def on_enter_accepted_round(self, event: object, state: State) -> None:
        evaluation = self.model.current_evaluation
        repair_outcome = self.model.current_repair_outcome

        if evaluation is None:
            self.raise_("advance")  # type: ignore[call-arg]
            return

        record = self._build_iteration_record(
            iteration=self.context.current_iteration,
            evaluation=evaluation,
            repair_outcome=repair_outcome,
        )
        self._record_iteration(
            record,
            pipeline_passed=repair_outcome.success if repair_outcome else None,
        )
        if repair_outcome and repair_outcome.success:
            score_val = record.metrics.get("score")
            self.context.prev_score = (
                float(score_val) if isinstance(score_val, (int, float)) else None
            )
            self.context.prev_classified = PipelineClassification()

        self.context.current_iteration += 1
        self.raise_("advance")  # type: ignore[call-arg]

    def _run_repair_phase(
        self,
        repo_root: Path,
        evaluation: EvaluationResult,
        feedback: FeedbackPackage,
        iteration_span: SpanHandle | None,
    ) -> RepairOutcome:
        repair_ctx = RepairContext(
            repo_root=repo_root,
            evaluation=evaluation,
            feedback=feedback,
            max_repair_attempts=self.max_policy_repairs,
            parent_trace=iteration_span,
            writer_model=self.deps.get_writer_model(),
            pipeline=self.deps.default_pipeline(),
        )
        repair_model = RepairMachineModel(
            context=repair_ctx,
            deps=self.deps,
        )
        repair_machine = RepairStateMachine(repair_model)
        return repair_machine.run()

    def _build_iteration_record(
        self,
        iteration: int,
        evaluation: EvaluationResult,
        repair_outcome: RepairOutcome | None,
    ) -> IterationRecord:
        metrics_map = evaluation.metrics.as_map()
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
            metrics=EvaluationMetrics.model_validate(metrics_map),
            pipeline_passed=repair_outcome.success if repair_outcome else None,
            pipeline_feedback=pipeline_feedback,
            repair_attempts=repair_attempts_val
            if isinstance(repair_attempts_val, int)
            else (repair_outcome.attempts_used if repair_outcome else None),
            max_policy_repairs=max_repairs_val if isinstance(max_repairs_val, int) else self.max_policy_repairs,
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

    def _record_iteration(
        self,
        record: IterationRecord,
        pipeline_passed: bool | None = None,
    ) -> None:
        pipeline_val = (
            record.pipeline_passed if pipeline_passed is None else pipeline_passed
        )
        record.pipeline_passed = pipeline_val
        outcome = getattr(self.model, "current_repair_outcome", None)
        if outcome and pipeline_val is False:
            if getattr(outcome, "pipeline_results", None):
                classified = outcome.pipeline_feedback or self.deps.classify_results(outcome.pipeline_results)
                record.pipeline_feedback = record.pipeline_feedback or classified
                failed_step = self._first_failed_step(outcome.pipeline_results)
                if failed_step:
                    record.failed_step = record.failed_step or failed_step.name
                    record.failed_exit_code = record.failed_exit_code or failed_step.exit_code
                    record.failed_summary = record.failed_summary or failed_step.stderr or failed_step.stdout
        if pipeline_val is False:
            if outcome and getattr(outcome, "pipeline_results", None):
                classified = outcome.pipeline_feedback or self.deps.classify_results(outcome.pipeline_results)
                record.pipeline_feedback = classified
            if record.failed_step is None:
                if outcome and getattr(outcome, "pipeline_results", None):
                    failed_step = self._first_failed_step(outcome.pipeline_results)
                    if failed_step:
                        record.failed_step = failed_step.name
                        record.failed_exit_code = failed_step.exit_code
                        record.failed_summary = failed_step.stderr or failed_step.stdout
                if record.failed_step is None:
                    record.failed_step = "pipeline_failed"
        if record.pipeline_feedback is None and outcome and getattr(outcome, "pipeline_results", None):
            record.pipeline_feedback = self.deps.classify_results(outcome.pipeline_results)
        if record.writer_error is None and outcome:
            if isinstance(outcome.writer_error, str):
                record.writer_error = outcome.writer_error
            elif getattr(outcome, "success", None) is False:
                record.writer_error = "writer_failed"
        self.context.history.append(record)

    def _first_failed_step(
        self, steps: Sequence[StepResult]
    ) -> StepResult | None:
        for step in steps:
            if step.exit_code != 0:
                return step
        return None


def _runtime_collaborators() -> dependency_wiring.RuntimeCollaborators:
    def find_runtime_repo_root() -> Path:
        return optimization_repo_root() or repo_root_utils.find_repo_root()

    return dependency_wiring.RuntimeCollaborators(
        index_folder=index_folder_tool.index_folder,
        evaluate_localization_batch=policy_evaluation.evaluate_localization_batch,
        load_tests=test_loader.load_tests,
        format_tests_for_prompt=test_loader.format_tests_for_prompt,
        build_frontier_context=type_loader.build_frontier_context,
        format_frontier_context=type_loader.format_frontier_context,
        compute_diff=diff_utils.compute_diff,
        format_diff=diff_utils.format_diff,
        interpret_diff=diff_utils.interpret_diff,
        generate_policy=agent_writer.generate_policy,
        emit_score=score_emitter.emit_score,
        classify_results=pipeline_module.classify_results,
        read_policy=policy_evaluation._read_policy,
        write_policy=policy_writer._write_policy,
        get_writer_model=env_utils.get_writer_model,
        start_observation=tracing.start_observation,
        find_repo_root=find_runtime_repo_root,
        default_pipeline=pipeline_module.default_pipeline,
    )


def build_loop_dependencies() -> LoopDependencies:
    return dependency_wiring.build_loop_dependencies(
        _runtime_collaborators(),
        evaluate_policy_on_item=policy_evaluation.evaluate_policy_on_item,
        build_iteration_feedback=feedback_wiring.build_iteration_feedback,
        attempt_policy_generation=policy_writer.attempt_policy_generation,
    )


def _fallback_history(
    opt_model: OptimizationMachineModel,
    parent_trace: SpanHandle | None,
) -> list[IterationRecord]:
    repo_root = repo_root_utils.find_repo_root()
    pipeline = pipeline_module.default_pipeline()
    fallback_parent = opt_model.context.run_trace or parent_trace
    pipeline_results, pipeline_success = dependency_wiring.run_policy_pipeline(
        pipeline,
        repo_root,
        repair_obs=fallback_parent,
        allow_no_parent=fallback_parent is None,
        emit_score=score_emitter.emit_score,
    )
    classified = pipeline_module.classify_results(pipeline_results)
    failed_step = dependency_wiring.first_failed_step(pipeline_results)
    metrics_map: dict[str, float | int | bool | None] = {}
    return [
        IterationRecord(
            iteration=0,
            metrics=EvaluationMetrics.model_validate(metrics_map),
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


def _baseline_record_model(
    baseline_record: "EvaluationRecord | Mapping[str, object] | None",
) -> "EvaluationRecord | None":
    if isinstance(baseline_record, Mapping):
        from harness.telemetry.hosted.baselines import (
            EvaluationRecord as EvaluationRecordModel,
        )

        return EvaluationRecordModel.model_validate(baseline_record)
    return baseline_record


def run_loop(
    task: LCATask,
    iterations: int = 5,
    parent_trace: SpanHandle | None = None,
    baseline_record: "EvaluationRecord | Mapping[str, object] | None" = None,
    session_id: str | None = None,
) -> list[IterationRecord]:
    """
    Closed-loop optimizer: execute, score, rewrite policy, repeat.
    """
    root_meta = RunMetadata(task=task, iterations=iterations, session_id=session_id)
    opt_model: OptimizationMachineModel | None = None
    history: list[IterationRecord] = []
    with tracing.start_observation(
        name="run_loop",
        parent=parent_trace,
        input=task.model_dump(),
        metadata=root_meta.model_dump(),
    ) as run_span:
        with tracing.propagate_context(
            trace_name="run_loop",
            session_id=session_id,
            metadata={"iterations": iterations, "session_id": session_id},
        ):
            context = LoopContext(
                task=task,
                iterations=iterations,
                session_id=session_id,
                parent_trace=parent_trace,
                run_metadata=root_meta,
                baseline_record=_baseline_record_model(baseline_record),
                run_trace=run_span,
            )
            opt_model = OptimizationMachineModel(
                context=context,
                deps=build_loop_dependencies(),
                max_policy_repairs=_MAX_POLICY_REPAIRS,
            )
            machine = OptimizationStateMachine(opt_model)
            try:
                history = machine.run()
            finally:
                tracing.flush_langfuse()
    if not history and opt_model is not None:
        history = _fallback_history(opt_model, parent_trace)
    return history
