from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List

import pytest
from statemachine.event import BoundEvent

from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.orchestration import feedback as feedback_module
from harness.orchestration.machine import OptimizationStateMachine, RepairStateMachine
from harness.orchestration.types import (
    AcceptedPolicyMeta,
    EvaluationMetrics,
    EvaluationResult,
    FailedRepairDetails,
    FeedbackPackage,
    IterationTasks,
    ICResult,
    IterationRecord,
    JCResult,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
)
from harness.prompts import build_writer_optimization_brief
from harness.pipeline.types import PipelineClassification, StepResult
from harness.utils.type_loader import FrontierContext


class RecordingSpan:
    def __init__(self, name: str | None = None, metadata: dict[str, object] | None = None) -> None:
        self.name = name
        self.metadata = metadata or {}
        self.ended: list[dict[str, Any]] = []

    def end(self, **kwargs: Any) -> None:
        self.ended.append(dict(kwargs))

    def update(self, **kwargs: Any) -> None:
        meta = kwargs.get("metadata")
        if isinstance(meta, dict):
            self.ended.append(dict(meta))
        else:
            self.ended.append(dict(kwargs))

    def __enter__(self) -> "RecordingSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class EmptyPipeline:
    def run(self, repo_root: Path, observation: object | None = None, allow_no_parent: bool = False) -> list[StepResult]:
        return []


def _make_feedback() -> FeedbackPackage:
    scorer_ctx = FrontierContext(signature="", graph_models="", types="", examples="", notes="")
    return FeedbackPackage(
        tests="",
        frontier_context="",
        frontier_context_details=scorer_ctx,
        optimization_brief=build_writer_optimization_brief(guidance=["hint"]),
    )


def _make_eval() -> EvaluationResult:
    return EvaluationResult(
        metrics=EvaluationMetrics.model_validate({"score": 0.0}),
        ic_result=ICResult(),
        jc_result=JCResult(),
        jc_metrics=EvaluationMetrics(),
        comparison_summary=None,
        policy_code="policy",
    )


def _task() -> LCATask:
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    context = LCAContext(issue_title="bug", issue_body="details")
    return LCATask(
        identity=identity,
        context=context,
        gold=LCAGold(changed_files=["a.py"]),
        repo="repo",
    )


def test_prepared_tasks_emit_models() -> None:
    base = _task()
    prepared = PreparedTasks(task=base, resolved_repo_path="resolved", jc_repo_id="jc-id")
    iteration_tasks = prepared.build_iteration_tasks()
    assert isinstance(iteration_tasks, IterationTasks)
    assert isinstance(iteration_tasks.task, LCATask)
    assert iteration_tasks.task.repo == "resolved"
    assert iteration_tasks.jc_repo_id == "jc-id"


def test_build_iteration_feedback_produces_typed_writer_brief(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frontier = FrontierContext(signature="sig", graph_models="", types="", examples="", notes="")
    monkeypatch.setattr(feedback_module, "load_tests", lambda repo_root: object())
    monkeypatch.setattr(feedback_module, "format_tests_for_prompt", lambda tests: "tests")
    monkeypatch.setattr(feedback_module, "build_frontier_context", lambda repo_root: frontier)
    monkeypatch.setattr(feedback_module, "format_frontier_context", lambda ctx: "frontier")
    evaluation = EvaluationResult(
        metrics=EvaluationMetrics.model_validate(
            {"score": 0.25, "iteration_regression": True, "additional_metrics": [{"name": "gold_hop", "value": 0.5}]}
        ),
        ic_result=ICResult(),
        jc_result=JCResult(),
        jc_metrics=EvaluationMetrics(),
        comparison_summary="baseline comparison",
        policy_code="policy",
        error="runner failed",
        success=False,
    )

    feedback = feedback_module.build_iteration_feedback(evaluation, None, None, tmp_path)

    assert feedback.tests == "tests"
    assert feedback.frontier_context == "frontier"
    assert feedback.optimization_brief.comparison.summary == "baseline comparison"
    assert feedback.optimization_brief.guidance.mode == "repair"
    assert any(finding.kind == "evaluation_error" and finding.message == "runner failed" for finding in feedback.optimization_brief.findings)
    assert any(finding.kind == "metric" and finding.value == 0.25 for finding in feedback.optimization_brief.findings)


def _repair_deps(
    writer_ok_sequence: List[bool], pipeline_ok_sequence: List[bool]
) -> tuple[LoopDependencies, list[RecordingSpan], list[object], list[tuple[str, object]]]:
    writer_seq = list(writer_ok_sequence)
    pipeline_seq = list(pipeline_ok_sequence)
    spans: list[RecordingSpan] = []
    seen_parent: list[object] = []
    events: list[tuple[str, object]] = []
    written: list[str] = []

    def attempt_policy_generation(**kwargs: Any) -> tuple[str | None, str | None]:
        parent_trace = kwargs.get("parent_trace")
        seen_parent.append(parent_trace)
        events.append(("writer", parent_trace))
        ok = writer_seq.pop(0) if writer_seq else False
        if ok:
            return "new_policy", None
        return None, "writer_failed"

    def run_policy_pipeline(*args: Any, **kwargs: Any) -> tuple[list[StepResult], bool]:
        ok = pipeline_seq.pop(0) if pipeline_seq else False
        result = StepResult(name="step", success=ok, exit_code=0 if ok else 1, stdout="", stderr="")
        return [result], ok

    def finalize_successful_policy(**kwargs: Any) -> tuple[str, AcceptedPolicyMeta]:
        return (
            kwargs.get("new_code", "new_policy"),
            AcceptedPolicyMeta(
                accepted_policy_source="post_pipeline_disk",
                accepted_policy_changed_by_pipeline=False,
                repair_attempts=kwargs.get("attempts_used", 0),
                max_policy_repairs=kwargs.get("max_repair_attempts", 0),
            ),
        )

    def finalize_failed_repair(**kwargs: Any) -> FailedRepairDetails:
        return FailedRepairDetails(
            failure_context="ctx",
            policy_code="policy",
            failed_step="step",
            failed_exit_code=1,
            failed_summary="failed",
            error="pipeline_failed",
        )

    @contextmanager
    def start_observation(**kwargs: Any) -> Iterator[RecordingSpan]:
        span = RecordingSpan(
            name=kwargs.get("name") if isinstance(kwargs.get("name"), str) else None,
            metadata=kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else None,
        )
        spans.append(span)
        events.append(("span_open", span))
        yield span

    deps = LoopDependencies(
        prepare_iteration_tasks=lambda task, trace=None: PreparedTasks(
            task=task,
            resolved_repo_path=".",
            jc_repo_id=".",
        ),
        evaluate_policy_on_item=lambda *a, **k: _make_eval(),  # type: ignore[arg-type]
        build_iteration_feedback=lambda *a, **k: _make_feedback(),  # type: ignore[arg-type]
        attempt_policy_generation=attempt_policy_generation,
        run_policy_pipeline=run_policy_pipeline,
        finalize_successful_policy=finalize_successful_policy,
        finalize_failed_repair=finalize_failed_repair,
        classify_results=lambda results: PipelineClassification(),
        format_pipeline_failure_context=lambda *a, **k: "failure",
        read_policy=lambda: "policy",
        write_policy=lambda code: written.append(code),
        get_writer_model=lambda: "model",
        start_observation=start_observation,
        find_repo_root=lambda: Path("."),
        default_pipeline=lambda: EmptyPipeline(),
    )
    return deps, spans, seen_parent, events


def _repair_spans(spans: list[RecordingSpan]) -> list[RecordingSpan]:
    return [span for span in spans if (span.name or "").startswith("policy_repair_attempt_")]


def test_repair_exhausts_after_failures() -> None:
    deps, spans, seen_parent, events = _repair_deps(writer_ok_sequence=[False, False], pipeline_ok_sequence=[])
    ctx = RepairContext(
        repo_root=Path("."),
        evaluation=_make_eval(),
        feedback=_make_feedback(),
        max_repair_attempts=2,
        parent_trace=None,
        writer_model="model",
    )
    model = RepairMachineModel(context=ctx, deps=deps)

    outcome = RepairStateMachine(model).run()

    assert outcome.success is False
    assert outcome.attempts_used == 2
    assert outcome.error in {"writer_failed", "pipeline_failed"}
    repair_spans = _repair_spans(spans)
    assert len(repair_spans) == 2
    assert len(seen_parent) == 2
    assert all(getattr(seen_parent[i], "name", None) == "writer_generation" for i in range(2))
    # Span should be ended with failed status
    assert all(span.ended for span in repair_spans)
    assert any(meta.get("status") == "failed" for span in repair_spans for meta in span.ended)
    # Ordering: span_open precedes writer per attempt
    assert events[0][0] == "span_open"
    assert any(event[0] == "writer" for event in events[1:])


def test_repair_succeeds_on_second_attempt() -> None:
    deps, spans, seen_parent, events = _repair_deps(writer_ok_sequence=[False, True], pipeline_ok_sequence=[True])
    ctx = RepairContext(
        repo_root=Path("."),
        evaluation=_make_eval(),
        feedback=_make_feedback(),
        max_repair_attempts=3,
        parent_trace=None,
        writer_model="model",
    )
    model = RepairMachineModel(context=ctx, deps=deps)

    outcome = RepairStateMachine(model).run()

    assert outcome.success is True
    assert outcome.attempts_used == 2
    assert outcome.accepted_policy_source == "post_pipeline_disk"
    assert outcome.failed_step is None
    repair_spans = _repair_spans(spans)
    assert len(repair_spans) == 2
    assert len(seen_parent) == 2
    assert all(getattr(seen_parent[i], "name", None) == "writer_generation" for i in range(2))
    assert any(meta.get("status") == "passed" for span in repair_spans for meta in span.ended)
    # Ordering: first attempt failed, second passed; each writer sees its own span
    assert events[0][0] == "span_open"
    assert any(event[0] == "writer" for event in events[1:])


def test_repair_writer_loop_uses_writer_session_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import harness.orchestration.machine as machine_module

    propagated: list[dict[str, object]] = []

    @contextmanager
    def propagate_cm(**kwargs: object):
        propagated.append(dict(kwargs))
        yield

    monkeypatch.setattr(machine_module.tracing, "propagate_context", propagate_cm)
    deps, spans, _seen_parent, _events = _repair_deps(
        writer_ok_sequence=[True],
        pipeline_ok_sequence=[True],
    )
    ctx = RepairContext(
        repo_root=Path("."),
        evaluation=_make_eval(),
        feedback=_make_feedback(),
        max_repair_attempts=1,
        parent_trace=None,
        writer_model="model",
        optimize_run_id="opt-run-1",
        writer_session_id="writer:opt-run-1:1:abc",
        task_identity="task-1",
        task_ordinal=1,
        task_total=2,
    )
    model = RepairMachineModel(context=ctx, deps=deps)

    outcome = RepairStateMachine(model).run()

    assert outcome.success is True
    assert propagated
    assert all(call.get("session_id") == "writer:opt-run-1:1:abc" for call in propagated)
    repair_span = _repair_spans(spans)[0]
    assert repair_span.metadata["writer_session_id"] == "writer:opt-run-1:1:abc"
    assert repair_span.metadata["optimize_run_id"] == "opt-run-1"
    writer_span = next(span for span in spans if span.name == "writer_generation")
    assert writer_span.metadata["session_id"] == "writer:opt-run-1:1:abc"
    pipeline_span = next(span for span in spans if span.name == "pipeline_check")
    assert pipeline_span.metadata["session_id"] == "writer:opt-run-1:1:abc"


def test_pipeline_failure_propagates_failure_details() -> None:
    deps, spans, seen_parent, events = _repair_deps(writer_ok_sequence=[True], pipeline_ok_sequence=[False])
    ctx = RepairContext(
        repo_root=Path("."),
        evaluation=_make_eval(),
        feedback=_make_feedback(),
        max_repair_attempts=1,
        parent_trace=None,
        writer_model="model",
    )
    model = RepairMachineModel(context=ctx, deps=deps)

    outcome = RepairStateMachine(model).run()

    assert outcome.success is False
    assert outcome.failed_step == "step"
    assert outcome.failed_exit_code == 1
    assert outcome.failed_summary == "failed"
    repair_spans = _repair_spans(spans)
    assert len(repair_spans) == 1
    assert len(seen_parent) == 1
    assert getattr(seen_parent[0], "name", None) == "writer_generation"
    assert any(meta.get("status") == "failed" for span in repair_spans for meta in span.ended)
    assert events[0][0] == "span_open"
    assert any(event[0] == "writer" for event in events[1:])


def test_repair_listener_uses_event_data_without_direct_event_key_inspection() -> None:
    assert "key" in vars(BoundEvent)
    root = Path(__file__).resolve().parents[1]
    assert "event.key" not in (root / "orchestration" / "listeners.py").read_text(encoding="utf-8")
    deps, spans, seen_parent, events = _repair_deps(writer_ok_sequence=[True], pipeline_ok_sequence=[True])
    ctx = RepairContext(
        repo_root=Path("."),
        evaluation=_make_eval(),
        feedback=_make_feedback(),
        max_repair_attempts=1,
        parent_trace=None,
        writer_model="model",
    )
    model = RepairMachineModel(context=ctx, deps=deps)

    outcome = RepairStateMachine(model).run()

    assert outcome.success is True
    repair_spans = _repair_spans(spans)
    assert len(repair_spans) == 1
    assert all(getattr(parent, "name", None) == "writer_generation" for parent in seen_parent)
    assert events[0][0] == "span_open"
    assert any(event[0] == "writer" for event in events[1:])


def test_failed_candidate_not_marked_accepted() -> None:
    deps, spans, _, events = _repair_deps(writer_ok_sequence=[False, False], pipeline_ok_sequence=[])
    ctx = RepairContext(
        repo_root=Path("."),
        evaluation=_make_eval(),
        feedback=_make_feedback(),
        max_repair_attempts=2,
        parent_trace=None,
        writer_model="model",
    )
    model = RepairMachineModel(context=ctx, deps=deps)

    outcome = RepairStateMachine(model).run()

    assert outcome.success is False
    assert outcome.accepted_policy_source is None
    assert outcome.accepted_policy_changed_by_pipeline is None
    assert any(meta.get("status") == "failed" for span in _repair_spans(spans) for meta in span.ended)
    assert events[0][0] == "span_open"
    assert any(event[0] == "writer" for event in events[1:])


def _opt_deps(ctx: LoopContext) -> LoopDependencies:
    span = RecordingSpan()

    def prepare_iteration_tasks(task: LCATask, trace: object | None = None) -> PreparedTasks:
        return PreparedTasks(
            task=task.with_repo("."),
            resolved_repo_path=".",
            jc_repo_id=".",
        )

    def evaluate_policy_on_item(
        task: LCATask,
        baseline,
        iteration_span: object | None = None,
        iteration_index: int | None = None,
    ) -> EvaluationResult:
        metrics = EvaluationMetrics.model_validate({"score": 1.0})
        record = IterationRecord(iteration=iteration_index or 0, metrics=metrics, pipeline_passed=True)
        ctx.history.append(record)
        return EvaluationResult(
            metrics=metrics,
            ic_result=ICResult(),
            jc_result=JCResult(),
            jc_metrics=EvaluationMetrics(),
            comparison_summary=None,
            policy_code="policy",
        )

    deps = LoopDependencies(
        prepare_iteration_tasks=prepare_iteration_tasks,
        evaluate_policy_on_item=evaluate_policy_on_item,
        build_iteration_feedback=lambda *a, **k: _make_feedback(),
        attempt_policy_generation=lambda **k: ("policy", None),
        run_policy_pipeline=lambda *a, **k: ([StepResult(name="step", success=True, exit_code=0, stdout="", stderr="")], True),
        finalize_successful_policy=lambda **k: ("policy", AcceptedPolicyMeta(accepted_policy_source="disk", accepted_policy_changed_by_pipeline=False, repair_attempts=0, max_policy_repairs=0)),
        finalize_failed_repair=lambda **k: FailedRepairDetails(failure_context="", policy_code="policy", failed_step=None, failed_exit_code=None, failed_summary=None, error=""),
        classify_results=lambda results: PipelineClassification(),
        format_pipeline_failure_context=lambda *a, **k: "",
        read_policy=lambda: "policy",
        write_policy=lambda code: None,
        get_writer_model=lambda: "model",
        start_observation=lambda **k: span,  # RecordingSpan is a context manager
        find_repo_root=lambda: Path("."),
        default_pipeline=lambda: EmptyPipeline(),
    )
    return deps


def test_optimization_completes_and_records_history() -> None:
    ctx = LoopContext(
        task=_task(),
        iterations=1,
        session_id=None,
        baseline_record=None,
        run_trace=RecordingSpan(),
        history=[],
        prepared_tasks=None,
    )
    deps = _opt_deps(ctx)
    model = OptimizationMachineModel(context=ctx, deps=deps, max_policy_repairs=1)
    sm = OptimizationStateMachine(model)
    sm.model.prep_ok = True  # bypass prep guard

    sm.run()

    assert ctx.history, "history should be recorded"
    assert ctx.history[0].metrics.get("score") == 1.0
    assert sm.current_state.id == "completed"
