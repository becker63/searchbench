from __future__ import annotations

from typing import Any, cast

from statemachine import State

from .loop_types import IterationRecord, LoopDependencies


def _safe_end_span(span: object | None, **kwargs) -> None:
    """Best-effort guard for Langfuse span termination."""
    if span is None or not hasattr(span, "end"):
        return
    try:
        cast(Any, span).end(**kwargs)
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
            ctx.repair_observation = machine.model.deps.start_span(
                ctx.parent_trace,
                f"policy_repair_attempt_{ctx.attempts_used}",
                metadata={"attempt": ctx.attempts_used},
            )

    def on_enter_retrying(self, event: object, state: State, event_data, **kwargs) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        self._end_attempt(machine, status="retrying")

    def on_enter_candidate_valid(self, event: object, state: State, event_data, **kwargs) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        self._end_attempt(machine, status="passed")

    def on_enter_repair_exhausted(self, event: object, state: State, event_data, **kwargs) -> None:
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

    def on_enter_evaluating(self, event: object, state: State, event_data, **kwargs) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        ctx = machine.model.context  # type: ignore[attr-defined]
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
        machine = cast(Any, getattr(event_data, "machine", None))
        self._finalize_iteration(machine, status="complete")

    def on_enter_completed(self, event: object, state: State, event_data, **kwargs) -> None:
        machine = cast(Any, getattr(event_data, "machine", None))
        self._finalize_iteration(machine, status="completed")

    def on_enter_failed(self, event: object, state: State, event_data, **kwargs) -> None:
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
        deps: LoopDependencies = cast(LoopDependencies, machine.deps)  # type: ignore[attr-defined]
        score_meta = {"iteration": record.iteration}
        if record.pipeline_passed is not None:
            deps.record_score(handle, "pipeline_passed", bool(record.pipeline_passed), score_meta)
        for key in ("score", "coverage_delta", "tool_error_rate"):
            value = record.metrics.get(key)
            if isinstance(value, (int, float, bool)):
                deps.record_score(handle, f"metrics.{key}", float(value), score_meta)
