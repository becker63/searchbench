from __future__ import annotations

"""
State machine listeners for tracing/metrics lifecycles.
No session creation; reacts to machine events only.
"""

import json
from typing import Any, ContextManager, Protocol

from statemachine import State

from harness.observability.score_emitter import emit_score_for_handle

from .loop_types import IterationRecord, OptimizationMachineModel, RepairMachineModel


class SpanHandle(Protocol):
    def end(self, **kwargs: object) -> None: ...

    def update(self, **kwargs: object) -> None: ...


class _EventTarget(Protocol):
    id: str


class _RepairEventData(Protocol):
    target: _EventTarget | None


class _OptEventData(Protocol):
    target: _EventTarget | None

class RepairTracingListener:
    """Listener that owns the full repair attempt span lifecycle.

    Opens spans via ``on_transition`` (which fires before the machine's
    ``on_enter_generating_candidate`` callback) so the span exists before
    ``_run_writer()`` does any work.  Closes spans on state-exit transitions
    (``retrying``, ``candidate_valid``, ``repair_exhausted``).
    """

    def on_transition(self, event: object, state: State, event_data: _RepairEventData, model: Any = None, **kwargs: object) -> None:
        target = getattr(event_data, 'target', None)
        if target is None or getattr(target, 'id', None) != "generating_candidate":
            return
        if model is None:
            return
        repair_model: RepairMachineModel = model
        ctx = repair_model.context
        if ctx.repair_observation is None:
            cm: ContextManager[SpanHandle] = repair_model.deps.start_observation(
                name=f"policy_repair_attempt_{ctx.attempts_used}",
                parent=ctx.parent_trace,
                metadata={"attempt": ctx.attempts_used},
            )
            ctx.repair_observation = cm.__enter__()
            ctx.repair_observation_cm = cm

    def on_enter_retrying(self, event: object, state: State, event_data: _RepairEventData, model: Any = None, **kwargs: object) -> None:
        if model is not None:
            self._end_attempt(model, status="retrying")

    def on_enter_candidate_valid(self, event: object, state: State, event_data: _RepairEventData, model: Any = None, **kwargs: object) -> None:
        if model is not None:
            self._end_attempt(model, status="passed")

    def on_enter_repair_exhausted(self, event: object, state: State, event_data: _RepairEventData, model: Any = None, **kwargs: object) -> None:
        if model is not None:
            self._end_attempt(model, status="failed")

    def _end_attempt(self, model: RepairMachineModel, status: str) -> None:
        ctx = model.context
        attempt_index = ctx.attempts_used - 1 if ctx.attempts_used > 0 else ctx.attempts_used
        metadata: dict[str, object] = {"status": status, "attempt": attempt_index}
        if ctx.pipeline_passed is False:
            metadata["pipeline_passed"] = False
        if ctx.error:
            metadata["error"] = ctx.error
        span = ctx.repair_observation
        if span:
            try:
                span.update(metadata=metadata)
            except Exception:
                pass
        cm = ctx.repair_observation_cm
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
        ctx.repair_observation = None
        ctx.repair_observation_cm = None


class OptimizationTracingListener:
    """Listener that owns iteration spans and summary metrics."""

    def __init__(self) -> None:
        self._last_recorded_iteration: int | None = None

    def on_enter_evaluating(self, event: object, state: State, event_data: _OptEventData, model: Any = None, **kwargs: object) -> None:
        if model is None:
            return
        opt_model: OptimizationMachineModel = model
        ctx = opt_model.context
        if ctx.prepared_tasks is None:
            opt_model.current_iteration_span = None
            opt_model.current_iteration_cm = None
            return
        iteration_tasks = ctx.prepared_tasks.build_iteration_tasks()
        cm: ContextManager[SpanHandle] = opt_model.deps.start_observation(
            name=f"iteration_{ctx.current_iteration}",
            parent=ctx.run_trace,
            metadata={
                "iteration": ctx.current_iteration,
                "iterations": ctx.iterations,
                "task": iteration_tasks.ic_task.model_dump(),
                "jc_repo": iteration_tasks.jc_task.repo,
            },
        )
        opt_model.current_iteration_span = cm.__enter__()
        opt_model.current_iteration_cm = cm

    def on_exit_accepted_round(self, event: object, state: State, event_data: _OptEventData, model: Any = None, **kwargs: object) -> None:
        if model is not None:
            self._finalize_iteration(model, status="complete")

    def on_enter_completed(self, event: object, state: State, event_data: _OptEventData, model: Any = None, **kwargs: object) -> None:
        if model is not None:
            self._finalize_iteration(model, status="completed")

    def on_enter_failed(self, event: object, state: State, event_data: _OptEventData, model: Any = None, **kwargs: object) -> None:
        if model is not None:
            self._finalize_iteration(model, status="failed")

    def _finalize_iteration(self, model: OptimizationMachineModel, status: str) -> None:
        ctx = model.context
        span = model.current_iteration_span
        record = ctx.history[-1] if ctx.history else None
        if record is not None and self._last_recorded_iteration == record.iteration:
            model.current_iteration_span = None
            model.current_iteration_cm = None
            return

        handle_obj: SpanHandle | None = span
        if handle_obj is None and ctx.run_trace is not None:
            handle_obj = ctx.run_trace
        if record is not None:
            self._record_iteration_scores(model, handle_obj, record)
            self._last_recorded_iteration = record.iteration

        if handle_obj is not None and span is not None:
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
            try:
                handle_obj.update(metadata=metadata)
            except Exception:
                pass
        cm = model.current_iteration_cm
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
        model.current_iteration_span = None
        model.current_iteration_cm = None

    def _record_iteration_scores(
        self,
        model: OptimizationMachineModel,
        handle: SpanHandle | None,
        record: IterationRecord,
    ) -> None:
        score_meta: dict[str, object] = {"iteration": record.iteration}
        comment: str | None = None
        try:
            comment = json.dumps(score_meta)
        except Exception:
            comment = str(score_meta)
        if record.pipeline_passed is not None:
            emit_score_for_handle(
                handle,
                name="pipeline_passed",
                value=bool(record.pipeline_passed),
                data_type="BOOLEAN",
                comment=comment,
            )
        for key in ("score",):
            value = record.metrics.get(key)
            if isinstance(value, (int, float, bool)):
                emit_score_for_handle(
                    handle,
                    name=f"metrics.{key}",
                    value=float(value),
                    data_type="NUMERIC" if not isinstance(value, bool) else "BOOLEAN",
                    comment=comment,
                )
