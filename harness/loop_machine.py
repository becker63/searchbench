from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

from statemachine import State, StateChart
from statemachine.event import BoundEvent

_BOUND_EVENT = cast(Any, BoundEvent)

# python-statemachine 3.x uses ``name`` on events while older callback dispatchers
# expect ``key``. Patch in a lightweight compatibility property so callbacks work
# uniformly without scattering conditionals.
if not hasattr(BoundEvent, "key"):  # pragma: no cover - defensive shim
    setattr(_BOUND_EVENT, "key", property(lambda self: self.name))  # type: ignore[attr-defined]

from .loop_listeners import OptimizationTracingListener, RepairTracingListener
from .loop_types import (
    AcceptedPolicy,
    EvaluationResult,
    FeedbackPackage,
    IterationRecord,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    RepairContext,
    RepairMachineModel,
    RepairOutcome,
)
from .pipeline.types import PipelineClassification, StepResult


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
        guidance = (
            "Repair mode: produce the smallest change that makes harness/policy.py pass ruff, basedpyright, and pytest. "
            "Add missing parameter annotations and return type -> float. Do not optimize score until the policy is valid."
            if ctx.attempts_used > 0
            else ctx.feedback.guidance_hint
        )
        if ctx.last_classified and ctx.last_classified.type_errors:
            guidance += " Fix all type errors and ensure parameters are annotated; avoid unknown types."

        new_code, writer_error = deps.attempt_policy_generation(
            feedback=ctx.feedback.feedback,
            initial_policy=ctx.evaluation.policy_code,
            tests=ctx.feedback.tests,
            scoring_context=ctx.feedback.scoring_context,
            feedback_str=ctx.feedback.feedback_str,
            guidance_hint=guidance,
            diff_str=ctx.feedback.diff_str,
            diff_hint=ctx.feedback.diff_hint,
            comparison_summary=ctx.feedback.comparison_summary,
            failure_context_str=ctx.failure_context,
            repair_attempt=ctx.attempts_used,
            parent_trace=ctx.repair_observation,
            pipeline_feedback=ctx.last_classified.model_dump() if ctx.last_classified else None,
        )

        if new_code is None:
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
        self.raise_("writer_generated")  # type: ignore[call-arg]
        self._run_pipeline()

    def _run_pipeline(self) -> None:
        ctx = self.context
        deps = self.deps
        pipeline_results, pipeline_success = deps.run_policy_pipeline(
            ctx.pipeline or deps.default_pipeline(),
            ctx.repo_root,
            repair_obs=ctx.repair_observation,
        )
        ctx.pipeline_results = pipeline_results
        ctx.pipeline_passed = pipeline_success
        ctx.attempts_used += 1

        if pipeline_success:
            final_policy, success_meta = deps.finalize_successful_policy(
                new_code=ctx.candidate_code or deps.read_policy(),
                attempts_used=ctx.attempts_used,
                max_repair_attempts=ctx.max_repair_attempts,
                repair_obs=ctx.repair_observation,
            )
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
            self.raise_("pipeline_retryable_failure")  # type: ignore[call-arg]
        else:
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
        super().__init__(
            model=model,
            listeners=[
                OptimizationTracingListener(),
            ],
        )

    # Entry point ---------------------------------------------------------
    def run(self) -> list[IterationRecord]:
        self._prepare_context()
        self.raise_("prepare")  # type: ignore[call-arg]
        return self.context.history

    # Guards --------------------------------------------------------------
    def prep_ready(self) -> bool:
        return self.model.prep_ok

    def evaluation_successful(self) -> bool:
        return bool(self.model.current_evaluation and self.model.current_evaluation.success)

    def has_more_iterations(self) -> bool:
        return self.context.current_iteration + 1 < self.context.iterations

    # Preparation ---------------------------------------------------------
    def _prepare_context(self) -> None:
        try:
            self.context.prepared_tasks = self.deps.prepare_iteration_tasks(
                self.context.task, self.context.run_trace
            )
            self.model.prep_ok = self.context.prepared_tasks is not None
        except Exception:
            self.context.prepared_tasks = None
            self.model.prep_ok = False
        if self.context.prepared_tasks is None and isinstance(self.context.task, dict):
            repo_val = self.context.task.get("repo")
            resolved_repo = repo_val if isinstance(repo_val, str) else None
            self.context.prepared_tasks = PreparedTasks(
                base_task=dict(self.context.task), resolved_repo_path=resolved_repo, jc_repo_id=None
            )
            self.model.prep_ok = True

    # Evaluation ----------------------------------------------------------
    def on_enter_evaluating(self, event: object, state: State) -> None:
        self.model.current_evaluation = None
        self.model.current_repair_outcome = None
        prepared_tasks = getattr(self.context, "prepared_tasks", None)
        if prepared_tasks is None:
            self.model.current_evaluation = None
            self.raise_("evaluation_done")  # type: ignore[call-arg]
            return

        ic_task, _ = prepared_tasks.build_iteration_tasks()
        self.model.current_evaluation = self.deps.evaluate_policy_on_item(
            ic_task,
            self.context.baseline_snapshot,
            self.model.current_iteration_span,
            self.context.current_iteration,
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
        feedback = self.deps.build_iteration_feedback(
            evaluation,
            self.context.prev_score,
            self.context.prev_classified,
            repo_root,
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
        iteration_span: object | None,
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
        metrics_map = dict(evaluation.metrics)
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
            metrics=metrics_map,
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
