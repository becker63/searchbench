"""
State machine listeners for tracing/metrics lifecycles.
No session creation; reacts to machine events only.
"""

from __future__ import annotations

from typing import Any, ContextManager, Protocol

from statemachine import State

from .types import RepairMachineModel


class SpanHandle(Protocol):
    def end(self, **kwargs: object) -> None: ...

    def update(self, **kwargs: object) -> None: ...


class _EventTarget(Protocol):
    id: str


class _RepairEventData(Protocol):
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
            metadata: dict[str, object] = {
                "attempt": ctx.attempts_used,
                "phase": "policy_repair_attempt",
            }
            if ctx.optimize_run_id:
                metadata["optimize_run_id"] = ctx.optimize_run_id
            if ctx.writer_session_id:
                metadata["writer_session_id"] = ctx.writer_session_id
                metadata["session_id"] = ctx.writer_session_id
            if ctx.task_identity:
                metadata["task_identity"] = ctx.task_identity
            if ctx.task_ordinal is not None:
                metadata["ordinal"] = ctx.task_ordinal
            if ctx.task_total is not None:
                metadata["total"] = ctx.task_total
            cm: ContextManager[SpanHandle] = repair_model.deps.start_observation(
                name=f"policy_repair_attempt_{ctx.attempts_used}",
                parent=ctx.parent_trace,
                metadata=metadata,
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
