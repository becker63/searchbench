from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from statemachine import State, StateChart
from statemachine.event import BoundEvent

_BOUND_EVENT = cast(Any, BoundEvent)

# python-statemachine 3.x uses ``name`` on events while older callback dispatchers
# expect ``key``. Patch in a lightweight compatibility property so callbacks work
# uniformly without scattering conditionals.
if not hasattr(BoundEvent, "key"):  # pragma: no cover - defensive shim
    setattr(_BOUND_EVENT, "key", property(lambda self: self.name))  # type: ignore[attr-defined]

from .loop_types import (
    AcceptedPolicy,
    EvaluationResult,
    FeedbackPackage,
    IterationRecord,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PipelineStepResult,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
    RepairOutcome,
)


def _safe_end_span(span: object | None, **kwargs) -> None:
    """Best-effort guard for Langfuse span termination."""
    if span is None or not hasattr(span, "end"):
        return
    try:
        cast(Any, span).end(**kwargs)
    except Exception:
        pass


class _TransitionTraceListener:
    """Lightweight listener to collect transition metadata for observability."""

    def __init__(self, label: str, sink: dict[str, list[str]] | None = None):
        self.label: str = label
        self.sink: dict[str, list[str]] = sink if sink is not None else {}

    def on_transition(self, event: object, state: State, event_data, **kwargs) -> None:
        machine = cast("StateChart", event_data.machine)
        trace_sink: dict[str, list[str]] = getattr(machine.model, "trace", self.sink)
        bucket = trace_sink.setdefault(self.label, [])
        bucket.append(f"{state.id}->{event_data.transition.target.id}")


class _RepairTracingListener:
    """Listener that manages repair attempt span lifecycle."""

    def on_enter_generating_candidate(self, event: object, state: State, event_data, **kwargs) -> None:
        machine: RepairStateMachine = cast(RepairStateMachine, event_data.machine)
        model = machine.model
        if not model.started:
            return
        ctx = model.context
        if ctx.repair_observation is None:
            ctx.repair_observation = model.deps.start_span(
                ctx.parent_trace,
                f"policy_repair_attempt_{ctx.attempts_used}",
                metadata={"attempt": ctx.attempts_used},
            )

    def on_enter_retrying(self, event: object, state: State, event_data, **kwargs) -> None:
        machine: RepairStateMachine = cast(RepairStateMachine, event_data.machine)
        self._end_attempt(machine, status="retrying")

    def on_enter_candidate_valid(self, event: object, state: State, event_data, **kwargs) -> None:
        machine: RepairStateMachine = cast(RepairStateMachine, event_data.machine)
        self._end_attempt(machine, status="passed")

    def on_enter_repair_exhausted(self, event: object, state: State, event_data, **kwargs) -> None:
        machine: RepairStateMachine = cast(RepairStateMachine, event_data.machine)
        self._end_attempt(machine, status="failed")

    def _end_attempt(self, machine: "RepairStateMachine", status: str) -> None:
        ctx = machine.model.context
        attempt_index = ctx.attempts_used - 1 if ctx.attempts_used > 0 else ctx.attempts_used
        metadata: dict[str, object] = {"status": status, "attempt": attempt_index}
        if ctx.pipeline_passed is False:
            metadata["pipeline_passed"] = False
        if ctx.metadata.get("error"):
            metadata["error"] = ctx.metadata["error"]
        _safe_end_span(ctx.repair_observation, metadata=metadata)
        ctx.repair_observation = None


class _OptimizationTracingListener:
    """Listener that owns iteration spans and summary metrics."""

    def __init__(self) -> None:
        self._last_recorded_iteration: int | None = None

    def on_enter_evaluating(self, event: object, state: State, event_data, **kwargs) -> None:
        machine: OptimizationStateMachine = cast(OptimizationStateMachine, event_data.machine)
        ctx = machine.model.context
        if ctx.prepared_tasks is None:
            machine.model.current_iteration_span = None  # noqa: SLF001
            return
        ic_task, jc_task = ctx.prepared_tasks.build_iteration_tasks()
        machine.model.current_iteration_span = machine.model.deps.start_span(  # noqa: SLF001
            ctx.run_trace,
            f"iteration_{ctx.current_iteration}",
            metadata={
                "iteration": ctx.current_iteration,
                "iterations": ctx.iterations,
                "task": dict(ic_task),
                "jc_repo": jc_task.get("repo"),
            },
        )

    def on_exit_accepted_round(self, event: object, state: State, event_data, **kwargs) -> None:
        machine: OptimizationStateMachine = cast(OptimizationStateMachine, event_data.machine)
        self._finalize_iteration(machine, status="complete")

    def on_enter_completed(self, event: object, state: State, event_data, **kwargs) -> None:
        machine: OptimizationStateMachine = cast(OptimizationStateMachine, event_data.machine)
        self._finalize_iteration(machine, status="completed")

    def on_enter_failed(self, event: object, state: State, event_data, **kwargs) -> None:
        machine: OptimizationStateMachine = cast(OptimizationStateMachine, event_data.machine)
        self._finalize_iteration(machine, status="failed")

    def _finalize_iteration(self, machine: "OptimizationStateMachine", status: str) -> None:
        ctx = machine.model.context
        span = machine.model.current_iteration_span
        record = ctx.history[-1] if ctx.history else None
        if record is not None and self._last_recorded_iteration == record.iteration:
            machine.model.current_iteration_span = None  # noqa: SLF001
            return

        handle = span or ctx.run_trace
        if record is not None:
            self._record_iteration_scores(machine, handle, record)
            self._last_recorded_iteration = record.iteration

        if span is not None:
            status_val = status
            if record and (record.error or record.writer_error or record.pipeline_passed is False):
                status_val = "failed"
            metadata: dict[str, object] = {
                "status": status_val,
                "iteration": record.iteration if record else ctx.current_iteration,
            }
            if record and record.pipeline_passed is not None:
                metadata["pipeline_passed"] = record.pipeline_passed
            failure_val = record.error if record else None
            failure_val = failure_val or (record.writer_error if record else None)
            if failure_val:
                metadata["error"] = failure_val
            _safe_end_span(span, metadata=metadata)
        machine.model.current_iteration_span = None  # noqa: SLF001

    def _record_iteration_scores(
        self,
        machine: "OptimizationStateMachine",
        handle: object | None,
        record: IterationRecord,
    ) -> None:
        deps = machine.deps
        score_meta = {"iteration": record.iteration}
        if record.pipeline_passed is not None:
            deps.record_score(handle, "pipeline_passed", bool(record.pipeline_passed), score_meta)
        for key in ("score", "coverage_delta", "tool_error_rate"):
            value = record.metrics.get(key)
            if isinstance(value, (int, float, bool)):
                deps.record_score(handle, f"metrics.{key}", float(value), score_meta)


class RepairStateMachine(StateChart[RepairMachineModel]):
    """
    Drives the candidate generation + pipeline repair loop using state-driven callbacks.
    """

    catch_errors_as_events = True

    generating_candidate = State(initial=True)
    running_pipeline = State()
    retrying = State()
    candidate_valid = State(final=True)
    repair_exhausted = State(final=True)

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
                _TransitionTraceListener("repair", model.trace),
                _RepairTracingListener(),
            ],
        )

    # Entry point ---------------------------------------------------------
    def run(self) -> RepairOutcome:
        self.model.started = True
        # Initialize repair-level observation once per machine run.
        ctx = self.context
        if ctx.repair_observation is None:
            ctx.repair_observation = self.deps.start_span(
                ctx.parent_trace,
                f"policy_repair_attempt_{ctx.attempts_used}",
                metadata={"attempt": ctx.attempts_used},
            )
        self._run_writer()
        attempts_used = self.context.attempts_used
        final_attempts = (
            self.context.max_repair_attempts
            if self.context.pipeline_passed is False
            else min(attempts_used, self.context.max_repair_attempts)
        )
        self.context.attempts_used = final_attempts
        self.context.metadata["repair_attempts"] = (
            max(0, final_attempts - 1)
            if self.context.pipeline_passed
            else self.context.max_repair_attempts
        )
        self.context.metadata["max_policy_repairs"] = self.context.max_repair_attempts
        return RepairOutcome(
            policy=self.deps.read_policy(),
            pipeline_results=self.context.pipeline_results,
            success=self.context.pipeline_passed,
            attempts_used=final_attempts,
            metadata=self.context.metadata,
        )

    # Writer/pipeline execution ------------------------------------------
    def on_enter_generating_candidate(self, event: object, state: State) -> None:
        if not self.model.started:
            return
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
        if ctx.last_classified and ctx.last_classified.get("type_errors"):
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
            pipeline_feedback=ctx.last_classified,
        )

        if new_code is None:
            ctx.metadata["writer_error"] = writer_error
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
        pipeline_results, raw_results, pipeline_success = deps.run_policy_pipeline(
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
            ctx.metadata.update(success_meta)
            self.raise_("pipeline_passed")  # type: ignore[call-arg]
            return

        classified = deps.classify_results(raw_results)
        ctx.last_classified = classified
        ctx.failure_context, ctx.evaluation.policy_code = deps.finalize_failed_repair(
            metadata=ctx.metadata,
            pipeline_results=[step.to_mapping() for step in pipeline_results],
            classified=classified,
            writer_model=ctx.writer_model,
            attempts_used=ctx.attempts_used,
        )
        if self._attempts_remaining():
            self.raise_("pipeline_retryable_failure")  # type: ignore[call-arg]
        else:
            self.raise_("pipeline_exhausted")  # type: ignore[call-arg]

    # Helpers -------------------------------------------------------------
    def _format_failure(self, pipeline_results, writer_error, attempt: int) -> str:
        return self.deps.format_pipeline_failure_context(
            [step.to_mapping() for step in pipeline_results],
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
                _TransitionTraceListener("optimization", model.trace),
                _OptimizationTracingListener(),
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
            self.model.prep_ok = False

    # Evaluation ----------------------------------------------------------
    def on_enter_evaluating(self, event, state) -> None:  # noqa: ANN001
        self.model.current_evaluation = None
        self.model.current_repair_outcome = None
        if self.context.prepared_tasks is None:
            self.model.current_evaluation = None
            self.raise_("evaluation_done")  # type: ignore[call-arg]
            return

        ic_task, jc_task = self.context.prepared_tasks.build_iteration_tasks()
        self.model.current_evaluation = self.deps.evaluate_policy_on_item(
            ic_task,
            self.context.baseline_snapshot,
            self.model.current_iteration_span,
            self.context.current_iteration,
        )
        self.raise_("evaluation_done")  # type: ignore[call-arg]

    # Repair --------------------------------------------------------------
    def on_enter_repairing(self, event, state) -> None:  # noqa: ANN001
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
    def on_enter_accepted_round(self, event, state) -> None:  # noqa: ANN001
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
            self.context.prev_classified = {
                "type_errors": "",
                "lint_errors": "",
                "test_failures": "",
                "passed": [],
            }

        self.context.current_iteration += 1
        self.raise_("advance")  # type: ignore[call-arg]

    def _run_repair_phase(
        self,
        repo_root: Path,
        evaluation: EvaluationResult,
        feedback: FeedbackPackage,
        iteration_span: object | None,
    ) -> RepairOutcome:
        if self.deps.repair_via_synthesize:
            outcome = self.deps.repair_via_synthesize(
                evaluation, feedback, repo_root, iteration_span
            )
            if outcome.pipeline_results and not isinstance(
                outcome.pipeline_results[0], PipelineStepResult
            ):
                outcome.pipeline_results = [
                    PipelineStepResult.from_mapping(r) for r in outcome.pipeline_results
                ]  # type: ignore[arg-type]
            return outcome
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
            deps=self._repair_dependencies(),
        )
        repair_machine = RepairStateMachine(repair_model)
        return repair_machine.run()

    def _build_iteration_record(
        self,
        iteration: int,
        evaluation: EvaluationResult,
        repair_outcome: RepairOutcome | None,
    ) -> IterationRecord:
        metadata = dict(repair_outcome.metadata) if repair_outcome else {}
        metrics_map = dict(evaluation.metrics)
        metrics_map.update(metadata)
        pipeline_feedback = metadata.get("pipeline_feedback")
        pipeline_feedback_map = (
            pipeline_feedback if isinstance(pipeline_feedback, Mapping) else None
        )
        repair_attempts_val = metadata.get("repair_attempts")
        max_repairs_val = metadata.get("max_policy_repairs")
        error_val = metadata.get("error")
        writer_err_val = metadata.get("writer_error")
        record = IterationRecord(
            iteration=iteration,
            metrics=metrics_map,
            pipeline_passed=repair_outcome.success
            if repair_outcome and isinstance(repair_outcome.success, bool)
            else None,
            pipeline_feedback=pipeline_feedback_map,
            repair_attempts=repair_attempts_val
            if isinstance(repair_attempts_val, int)
            else (repair_outcome.attempts_used if repair_outcome else None),
            max_policy_repairs=max_repairs_val
            if isinstance(max_repairs_val, int)
            else self.max_policy_repairs,
            error=error_val if isinstance(error_val, str) else None,
            writer_error=writer_err_val if isinstance(writer_err_val, str) else None,
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
        failed_summary_val = metadata.get("failed_summary")
        if isinstance(failed_summary_val, str) and not record.failed_summary:
            record.failed_summary = failed_summary_val
        if repair_outcome and repair_outcome.success:
            source_val = metadata.get("accepted_policy_source")
            changed_val = metadata.get("accepted_policy_changed_by_pipeline")
            record.accepted_policy = AcceptedPolicy(
                code=repair_outcome.policy,
                source=source_val if isinstance(source_val, str) else None,
                changed_by_pipeline=changed_val
                if isinstance(changed_val, bool)
                else None,
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
        self.context.history.append(record)

    def _first_failed_step(
        self, steps: Sequence[PipelineStepResult]
    ) -> PipelineStepResult | None:
        for step in steps:
            if isinstance(step.exit_code, int) and step.exit_code != 0:
                return step
            if step.exit_code is None and step.success is False:
                return step
        return None

    def _repair_dependencies(self) -> LoopDependencies:
        # Reuse the same dependency set for nested repair machine where possible.
        return LoopDependencies(
            prepare_iteration_tasks=self.deps.prepare_iteration_tasks,
            evaluate_policy_on_item=self.deps.evaluate_policy_on_item,
            build_iteration_feedback=self.deps.build_iteration_feedback,
            attempt_policy_generation=self.deps.attempt_policy_generation,
            run_policy_pipeline=self.deps.run_policy_pipeline,
            finalize_successful_policy=self.deps.finalize_successful_policy,
            finalize_failed_repair=self.deps.finalize_failed_repair,
            classify_results=self.deps.classify_results,
            format_pipeline_failure_context=self.deps.format_pipeline_failure_context,
            read_policy=self.deps.read_policy,
            write_policy=self.deps.write_policy,
            get_writer_model=self.deps.get_writer_model,
            record_score=self.deps.record_score,
            start_span=self.deps.start_span,
            find_repo_root=self.deps.find_repo_root,
            default_pipeline=self.deps.default_pipeline,
            repair_via_synthesize=None,
        )
