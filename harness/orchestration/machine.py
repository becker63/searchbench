"""
State machines and lifecycle for optimization/repair loops.
Holds state transitions and callbacks; no CLI or entrypoint concerns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jcodemunch_mcp.tools.index_folder as index_folder_tool
from langfuse import propagate_attributes
from statemachine import State, StateChart
from statemachine.event import BoundEvent

from harness import pipeline as pipeline_module
from harness.agents import writer as agent_writer
from harness.log import bind_logger, get_logger, short_task_label, tail_text
from harness.pipeline.types import StepResult
from harness.prompts import WriterGuidance
from harness.telemetry.tracing import serialize_observation_metadata
from harness.utils import diff as diff_utils
from harness.utils import env as env_utils
from harness.utils import repo_root as repo_root_utils
from harness.utils import test_loader, type_loader

from . import dependencies as dependency_wiring
from . import evaluation as policy_evaluation
from . import feedback as feedback_wiring
from . import writer as policy_writer
from .listeners import RepairTracingListener
from .runtime_context import optimization_repo_root
from .types import (
    LoopDependencies,
    RepairContext,
    RepairMachineModel,
    RepairOutcome,
)

_MAX_POLICY_REPAIRS = 8
_LOGGER = get_logger(__name__)

# The installed python-statemachine release still reads ``BoundEvent.key``
# internally while dispatching callbacks. Keep the patch isolated here; harness
# callbacks and listeners use state hooks and EventData instead.
if not hasattr(BoundEvent, "key"):  # pragma: no cover - depends on library release
    setattr(BoundEvent, "key", property(lambda self: self.name))  # type: ignore[attr-defined]


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
            pipeline_feedback=self.context.pipeline_feedback
            or self.context.last_classified,
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

    def _repair_logger(self):
        ctx = self.context
        iteration_ordinal = ctx.iteration + 1 if ctx.iteration is not None else None
        return bind_logger(
            _LOGGER,
            optimize_run_id=ctx.optimize_run_id,
            writer_session_id=ctx.writer_session_id,
            task=short_task_label(
                ctx.task_identity or "task",
                ordinal=ctx.task_ordinal,
                total=ctx.task_total,
            ),
            task_identity=ctx.task_identity,
            iteration=ctx.iteration,
            iteration_ordinal=iteration_ordinal,
            iterations=ctx.iterations,
        )

    def _attempt_logger(self):
        return bind_logger(
            self._repair_logger(),
            attempt=self.context.attempts_used,
            max_attempts=self.context.max_repair_attempts,
        )

    def _pipeline_classification_label(self) -> str | None:
        classified = self.context.last_classified
        if classified is None:
            return None
        labels: list[str] = []
        if classified.type_errors:
            labels.append("type_errors")
        if classified.lint_errors:
            labels.append("lint_errors")
        if classified.test_failures:
            labels.append("test_failures")
        if not labels and classified.passed:
            labels.append("passed_only")
        return "+".join(labels) if labels else None

    def _log_pipeline_failures(self) -> None:
        failure_logger = self._attempt_logger()
        classification = self._pipeline_classification_label()
        for step in self.context.pipeline_results:
            if step.exit_code == 0:
                continue
            failure_logger.warning(
                "pipeline_step_failed",
                step_name=step.name,
                command=" ".join(step.command) if step.command else None,
                exit_code=step.exit_code,
                classification=classification,
                stderr_tail=tail_text(step.stderr),
                stdout_tail=tail_text(step.stdout),
            )

    def _writer_correlation_metadata(self, *, phase: str) -> dict[str, object]:
        ctx = self.context
        metadata: dict[str, object] = {
            "phase": phase,
            "attempt": ctx.attempts_used,
        }
        if ctx.optimize_run_id:
            metadata["optimize_run_id"] = ctx.optimize_run_id
        if ctx.writer_session_id:
            metadata["writer_session_id"] = ctx.writer_session_id
            # Kept on writer-loop spans only so child score emission and tests
            # can correlate with the Langfuse session without putting evaluator
            # work into the same replay surface.
            metadata["session_id"] = ctx.writer_session_id
        if ctx.task_identity:
            metadata["task_identity"] = ctx.task_identity
        if ctx.task_ordinal is not None:
            metadata["ordinal"] = ctx.task_ordinal
        if ctx.task_total is not None:
            metadata["total"] = ctx.task_total
        return metadata

    def _run_writer(self) -> None:
        ctx = self.context
        deps = self.deps
        attempt_logger = self._attempt_logger()
        guidance = list(ctx.feedback.optimization_brief.guidance.instructions)
        mode = ctx.feedback.optimization_brief.guidance.mode
        if ctx.attempts_used > 0:
            mode = "repair"
            guidance.append(
                "Repair mode: produce the smallest change that makes harness/policy.py pass ruff, basedpyright, and pytest."
            )
            guidance.append(
                "Add missing parameter annotations and return type -> float before optimizing frontier_priority."
            )
        if ctx.last_classified and ctx.last_classified.type_errors:
            guidance.append(
                "Fix all type errors and ensure parameters are annotated; avoid unknown types."
            )
        optimization_brief = ctx.feedback.optimization_brief.model_copy(
            update={"guidance": WriterGuidance(mode=mode, instructions=guidance)}
        )

        attempt_logger.info(
            "writer_generation_started",
            writer_model=ctx.writer_model or "unknown",
            failure_context_chars=len(ctx.failure_context or ""),
        )
        writer_metadata = {
            **self._writer_correlation_metadata(phase="writer_generation"),
            "writer_model": ctx.writer_model,
            "failure_context_chars": len(ctx.failure_context or ""),
        }
        repair_span = ctx.repair_observation
        if repair_span is None:
            raise RuntimeError("policy repair span missing before writer_generation")
        with propagate_attributes(session_id=ctx.writer_session_id):
            with repair_span.start_as_current_observation(
                name="writer_generation",
                as_type="span",
                metadata=serialize_observation_metadata(writer_metadata),
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
                    writer_session_id=ctx.writer_session_id,
                    pipeline_feedback=ctx.last_classified,
                )
                try:
                    writer_span.update(
                        metadata=serialize_observation_metadata(
                            {
                                "attempt": ctx.attempts_used,
                                "code_returned": new_code is not None,
                                "writer_error": writer_error,
                                "writer_session_id": ctx.writer_session_id,
                            }
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    attempt_logger.warning(
                        "langfuse_metadata_update_failed",
                        operation="writer_generation.metadata",
                        observation="writer_generation",
                        keys=["attempt", "code_returned", "writer_error", "writer_session_id"],
                        error=str(exc),
                    )

        if new_code is None:
            attempt_logger.warning(
                "writer_generation_completed",
                code_returned=False,
                writer_error=writer_error,
            )
            ctx.writer_error = writer_error
            ctx.error = writer_error or "writer_failed"
            ctx.attempts_used += 1
            ctx.failure_context = self._format_failure(
                [], writer_error, ctx.attempts_used
            )
            if self._attempts_remaining():
                attempt_logger.warning(
                    "policy_candidate_rejected",
                    stage="writer_generation",
                    rejection_reason=ctx.error,
                )
                bind_logger(
                    self._repair_logger(),
                    attempt=ctx.attempts_used,
                    max_attempts=ctx.max_repair_attempts,
                ).info(
                    "policy_candidate_retrying",
                    previous_attempt=ctx.attempts_used - 1,
                    rejection_reason=ctx.error,
                )
                self.raise_("writer_retryable_failure")  # type: ignore[call-arg]
            else:
                attempt_logger.warning(
                    "policy_candidate_rejected",
                    stage="writer_generation",
                    rejection_reason=ctx.error,
                )
                attempt_logger.warning(
                    "policy_candidate_exhausted",
                    rejection_reason=ctx.error,
                )
                self.raise_("writer_exhausted")  # type: ignore[call-arg]
            return

        deps.write_policy(new_code)
        ctx.candidate_code = new_code
        attempt_logger.info(
            "writer_generation_completed",
            code_returned=True,
            code_chars=len(new_code),
            writer_session_id=ctx.writer_session_id,
        )
        self.raise_("writer_generated")  # type: ignore[call-arg]
        self._run_pipeline()

    def _run_pipeline(self) -> None:
        ctx = self.context
        deps = self.deps
        pipeline = ctx.pipeline or deps.default_pipeline()
        attempt_logger = self._attempt_logger()
        check_names = [
            getattr(step, "name", "unknown") for step in getattr(pipeline, "steps", [])
        ]
        attempt_logger.info(
            "pipeline_check_started",
            checks=check_names,
            repo_root=str(ctx.repo_root),
            writer_session_id=ctx.writer_session_id,
        )
        pipeline_metadata = {
            **self._writer_correlation_metadata(phase="pipeline_check"),
            "checks": check_names,
            "repo_root": str(ctx.repo_root),
        }
        repair_span = ctx.repair_observation
        if repair_span is None:
            raise RuntimeError("policy repair span missing before pipeline_check")
        with propagate_attributes(session_id=ctx.writer_session_id):
            with repair_span.start_as_current_observation(
                name="pipeline_check",
                as_type="span",
                metadata=serialize_observation_metadata(pipeline_metadata),
            ) as pipeline_span:
                pipeline_results, pipeline_success = deps.run_policy_pipeline(
                    pipeline,
                    ctx.repo_root,
                    repair_obs=pipeline_span,
                )
                try:
                    pipeline_span.update(
                        metadata=serialize_observation_metadata(
                            {
                                "pipeline_passed": pipeline_success,
                                "checks_run": [step.name for step in pipeline_results],
                                "writer_session_id": ctx.writer_session_id,
                            }
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    attempt_logger.warning(
                        "langfuse_metadata_update_failed",
                        operation="pipeline_check.metadata",
                        observation="pipeline_check",
                        keys=["checks_run", "pipeline_passed", "writer_session_id"],
                        error=str(exc),
                    )
        # Enforce canonical step result shape at the boundary.
        ctx.pipeline_results = [
            step if isinstance(step, StepResult) else StepResult.from_external(step)
            for step in pipeline_results
        ]
        ctx.pipeline_passed = pipeline_success
        ctx.attempts_used += 1
        attempt_logger.info(
            "pipeline_check_result",
            passed=pipeline_success,
            checks_run=[step.name for step in ctx.pipeline_results],
        )

        if pipeline_success:
            acceptance_metadata = {
                **self._writer_correlation_metadata(phase="policy_acceptance"),
                "attempts_used": ctx.attempts_used,
                "status": "accepted",
            }
            repair_span = ctx.repair_observation
            if repair_span is None:
                raise RuntimeError("policy repair span missing before policy_acceptance")
            with propagate_attributes(session_id=ctx.writer_session_id):
                with repair_span.start_as_current_observation(
                    name="policy_acceptance",
                    as_type="span",
                    metadata=serialize_observation_metadata(acceptance_metadata),
                ) as acceptance_span:
                    final_policy, success_meta = deps.finalize_successful_policy(
                        new_code=ctx.candidate_code or deps.read_policy(),
                        attempts_used=ctx.attempts_used,
                        max_repair_attempts=ctx.max_repair_attempts,
                        repair_obs=acceptance_span,
                    )
                    try:
                        acceptance_span.update(
                            metadata=serialize_observation_metadata(
                                {
                                    **success_meta.model_dump(mode="json"),
                                    "writer_session_id": ctx.writer_session_id,
                                }
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        attempt_logger.warning(
                            "langfuse_metadata_update_failed",
                            operation="policy_acceptance.metadata",
                            observation="policy_acceptance",
                            keys=sorted(
                                [
                                    *success_meta.model_dump(mode="json").keys(),
                                    "writer_session_id",
                                ]
                            ),
                            error=str(exc),
                        )
            ctx.candidate_code = final_policy
            ctx.accepted_policy_source = success_meta.accepted_policy_source
            ctx.accepted_policy_changed_by_pipeline = (
                success_meta.accepted_policy_changed_by_pipeline
            )
            ctx.repair_attempts_reported = success_meta.repair_attempts
            ctx.max_policy_repairs_reported = success_meta.max_policy_repairs
            bind_logger(
                self._repair_logger(),
                attempt=ctx.attempts_used - 1,
                max_attempts=ctx.max_repair_attempts,
            ).info(
                "policy_candidate_accepted",
                accepted_policy_source=ctx.accepted_policy_source,
                changed_by_pipeline=ctx.accepted_policy_changed_by_pipeline,
            )
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
        self._log_pipeline_failures()
        if self._attempts_remaining():
            attempt_logger.warning(
                "policy_candidate_rejected",
                stage="pipeline_check",
                failed_step=ctx.failed_step,
                exit_code=ctx.failed_exit_code,
                rejection_reason=ctx.error,
                classification=self._pipeline_classification_label(),
            )
            bind_logger(
                self._repair_logger(),
                attempt=ctx.attempts_used,
                max_attempts=ctx.max_repair_attempts,
            ).info(
                "policy_candidate_retrying",
                previous_attempt=ctx.attempts_used - 1,
                failed_step=ctx.failed_step,
                rejection_reason=ctx.error,
            )
            self.raise_("pipeline_retryable_failure")  # type: ignore[call-arg]
        else:
            attempt_logger.warning(
                "policy_candidate_rejected",
                stage="pipeline_check",
                failed_step=ctx.failed_step,
                exit_code=ctx.failed_exit_code,
                rejection_reason=ctx.error,
                classification=self._pipeline_classification_label(),
            )
            attempt_logger.warning(
                "policy_candidate_exhausted",
                failed_step=ctx.failed_step,
                exit_code=ctx.failed_exit_code,
                rejection_reason=ctx.error,
            )
            self.raise_("pipeline_exhausted")  # type: ignore[call-arg]

    # Helpers -------------------------------------------------------------
    def _format_failure(
        self, pipeline_results: list[StepResult], writer_error: str | None, attempt: int
    ) -> str:
        return self.deps.format_pipeline_failure_context(
            pipeline_results,
            self.context.last_classified,
            writer_error,
            self.context.writer_model,
            attempt,
        )


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
        classify_results=pipeline_module.classify_results,
        read_policy=policy_evaluation._read_policy,
        write_policy=policy_writer._write_policy,
        get_writer_model=env_utils.get_writer_model,
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
