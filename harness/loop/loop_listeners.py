from __future__ import annotations

"""
State machine listeners for tracing/metrics lifecycles.
No session creation; reacts to machine events only.
"""

import json
from typing import Any, Mapping, cast

from statemachine import State

from harness.observability.score_emitter import emit_score_for_handle

from .loop_types import IterationRecord


def _safe_end_span(span: object | None, **kwargs: object) -> None:
    """Best-effort guard for Langfuse span termination."""
    if span is None or not hasattr(span, "end"):
        return
    try:
        end_kwargs: dict[str, object] = {}
        metadata = kwargs.pop("metadata", None)
        if isinstance(metadata, Mapping):
            end_kwargs.update(dict(metadata))
        end_kwargs.update(kwargs)
        cast(Any, span).end(**end_kwargs)
    except Exception:
        pass


class RepairTracingListener:
    """Listener that owns the full repair attempt span lifecycle.

    Opens spans via ``on_transition`` (which fires before the machine's
    ``on_enter_generating_candidate`` callback) so the span exists before
    ``_run_writer()`` does any work.  Closes spans on state-exit transitions
    (``retrying``, ``candidate_valid``, ``repair_exhausted``).
    """

    def on_transition(self, event: object, state: State, event_data: object, **kwargs: object) -> None:
        target = getattr(event_data, "target", None)
        if target is None or getattr(target, "id", None) != "generating_candidate":
            return
        machine = cast(Any, getattr(event_data, "machine", None))
        ctx = machine.model.context  # type: ignore[attr-defined]
        if ctx.repair_observation is None:
            ctx.repair_observation = machine.model.deps.start_observation(
                name=f"policy_repair_attempt_{ctx.attempts_used}",
                parent=ctx.parent_trace,
                metadata={"attempt": ctx.attempts_used},
            )

    def on_enter_retrying(self, event: object, state: State, event_data: object, **kwargs: object) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        self._end_attempt(machine, status="retrying")

    def on_enter_candidate_valid(self, event: object, state: State, event_data: object, **kwargs: object) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        self._end_attempt(machine, status="passed")

    def on_enter_repair_exhausted(self, event: object, state: State, event_data: object, **kwargs: object) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        self._end_attempt(machine, status="failed")

    def _end_attempt(self, machine: Any, status: str) -> None:
        ctx = machine.model.context  # type: ignore[attr-defined]
        attempt_index = ctx.attempts_used - 1 if ctx.attempts_used > 0 else ctx.attempts_used
        metadata: dict[str, object] = {"status": status, "attempt": attempt_index}
        if ctx.pipeline_passed is False:
            metadata["pipeline_passed"] = False
        if getattr(ctx, "error", None):
            metadata["error"] = ctx.error  # type: ignore[attr-defined]
        _safe_end_span(ctx.repair_observation, metadata=metadata)
        ctx.repair_observation = None


class OptimizationTracingListener:
    """Listener that owns iteration spans and summary metrics."""

    def __init__(self) -> None:
        self._last_recorded_iteration: int | None = None

    def on_enter_evaluating(self, event: object, state: State, event_data: object, **kwargs: object) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        ctx = machine.model.context  # type: ignore[attr-defined]
        if ctx.prepared_tasks is None:
            machine.model.current_iteration_span = None  # noqa: SLF001
            return
        iteration_tasks = ctx.prepared_tasks.build_iteration_tasks()
        machine.model.current_iteration_span = machine.model.deps.start_observation(  # noqa: SLF001
            name=f"iteration_{ctx.current_iteration}",
            parent=ctx.run_trace,
            metadata={
                "iteration": ctx.current_iteration,
                "iterations": ctx.iterations,
                "task": iteration_tasks.ic_task.model_dump(),
                "jc_repo": iteration_tasks.jc_task.repo,
            },
        )

    def on_exit_accepted_round(self, event: object, state: State, event_data: object, **kwargs: object) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        self._finalize_iteration(machine, status="complete")

    def on_enter_completed(self, event: object, state: State, event_data: object, **kwargs: object) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        self._finalize_iteration(machine, status="completed")

    def on_enter_failed(self, event: object, state: State, event_data: object, **kwargs: object) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        self._finalize_iteration(machine, status="failed")

    def _finalize_iteration(self, machine: Any, status: str) -> None:
        ctx = machine.model.context  # type: ignore[attr-defined]
        span = machine.model.current_iteration_span  # type: ignore[attr-defined]
        record = ctx.history[-1] if ctx.history else None
        if record is not None and self._last_recorded_iteration == record.iteration:
            machine.model.current_iteration_span = None  # type: ignore[attr-defined]  # noqa: SLF001
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
        machine.model.current_iteration_span = None  # type: ignore[attr-defined]  # noqa: SLF001

    def _record_iteration_scores(
        self,
        machine: Any,
        handle: object | None,
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
