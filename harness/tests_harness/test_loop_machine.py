from __future__ import annotations

from pathlib import Path
from typing import Mapping

from harness.loop import (
    OptimizationStateMachine,
    RepairStateMachine,
    AcceptedPolicyMeta,
    FeedbackPackage,
    EvaluationResult,
    FailedRepairDetails,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
)
from harness.pipeline.types import PipelineClassification, StepResult


def _make_deps(**overrides) -> LoopDependencies:
    def prepare_iteration_tasks(task: Mapping[str, object], trace: object | None = None) -> PreparedTasks:
        return PreparedTasks(base_task=dict(task), resolved_repo_path=None, jc_repo_id=None)

    def evaluate_policy_on_item(
        ic_task: Mapping[str, object],
        baseline,
        iteration_span: object | None = None,
        iteration_index: int | None = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            metrics={"score": 1.0},
            ic_result={},
            jc_result={},
            jc_metrics={},
            comparison_summary=None,
            policy_code="policy",
        )

    def build_iteration_feedback(
        evaluation: EvaluationResult,
        prev_score: float | None,
        prev_classified,
        repo_root: Path,
    ) -> FeedbackPackage:
        return FeedbackPackage(
            tests="",
            scoring_context="",
            comparison_summary=None,
            feedback={"score": 1.0},
            feedback_str="",
            guidance_hint="hint",
            diff_str="",
            diff_hint="",
        )

    def attempt_policy_generation(**kwargs) -> tuple[str | None, str | None]:
        return ("policy", None)

    def run_policy_pipeline(pipeline: object, repo_root: Path, repair_obs: object | None = None) -> tuple[list[StepResult], bool]:
        result = StepResult(name="step", success=True, exit_code=0, stdout="", stderr="")
        return [result], True

    def finalize_successful_policy(**kwargs) -> tuple[str, AcceptedPolicyMeta]:
        return (
            "policy",
            AcceptedPolicyMeta(
                accepted_policy_source="post_pipeline_disk",
                accepted_policy_changed_by_pipeline=False,
                repair_attempts=0,
                max_policy_repairs=1,
            ),
        )

    def finalize_failed_repair(**kwargs) -> FailedRepairDetails:
        return FailedRepairDetails(
            failure_context="",
            policy_code="policy",
            failed_step=None,
            failed_exit_code=None,
            failed_summary=None,
            error="pipeline_failed",
        )

    def classify_results(results: list[StepResult]):
        return PipelineClassification()

    deps = LoopDependencies(
        prepare_iteration_tasks=prepare_iteration_tasks,
        evaluate_policy_on_item=evaluate_policy_on_item,
        build_iteration_feedback=build_iteration_feedback,
        attempt_policy_generation=attempt_policy_generation,
        run_policy_pipeline=run_policy_pipeline,
        finalize_successful_policy=finalize_successful_policy,
        finalize_failed_repair=finalize_failed_repair,
        classify_results=classify_results,
        format_pipeline_failure_context=lambda *a, **k: "",
        read_policy=lambda: "policy",
        write_policy=lambda code: None,
        get_writer_model=lambda: "model",
        start_observation=lambda *a, **k: None,
        find_repo_root=lambda: Path("."),
        default_pipeline=lambda: object(),
    )
    for key, val in overrides.items():
        setattr(deps, key, val)
    return deps


def _make_evaluation(
    *,
    metrics: dict[str, object] | None = None,
    ic_result: Mapping[str, object] | None = None,
    jc_result: Mapping[str, object] | None = None,
    jc_metrics: Mapping[str, object] | None = None,
    comparison_summary: str | None = None,
    policy_code: str = "policy",
    success: bool = True,
    error: str | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        metrics=metrics or {},
        ic_result=ic_result or {},
        jc_result=jc_result or {},
        jc_metrics=jc_metrics or {},
        comparison_summary=comparison_summary,
        policy_code=policy_code,
        success=success,
        error=error,
    )


def _make_feedback(
    *,
    tests: str = "",
    scoring_context: str = "",
    comparison_summary: str | None = None,
    feedback: dict[str, object] | None = None,
    feedback_str: str = "",
    guidance_hint: str = "",
    diff_str: str = "",
    diff_hint: str = "",
) -> FeedbackPackage:
    return FeedbackPackage(
        tests=tests,
        scoring_context=scoring_context,
        comparison_summary=comparison_summary,
        feedback=feedback or {},
        feedback_str=feedback_str,
        guidance_hint=guidance_hint,
        diff_str=diff_str,
        diff_hint=diff_hint,
    )


def _make_repair_ctx(
    deps: LoopDependencies,
    *,
    max_repair_attempts: int = 3,
    repo_root: Path | None = None,
    evaluation: EvaluationResult | None = None,
    feedback: FeedbackPackage | None = None,
    parent_trace=None,
    writer_model: str | None = "model",
) -> RepairContext:
    return RepairContext(
        repo_root=repo_root or Path("."),
        evaluation=evaluation or _make_evaluation(),
        feedback=feedback or _make_feedback(),
        max_repair_attempts=max_repair_attempts,
        parent_trace=parent_trace,
        writer_model=writer_model,
    )


def test_optimization_machine_success_path():
    deps = _make_deps()
    model = OptimizationMachineModel(
        context=LoopContext(task={"symbol": "s"}, iterations=1, baseline_snapshot=None, run_trace=None),
        deps=deps,
        max_policy_repairs=1,
    )
    machine = OptimizationStateMachine(model)
    history = machine.run()
    assert history and history[0].pipeline_passed is True
    assert history[0].accepted_policy and history[0].accepted_policy.source == "post_pipeline_disk"
    state = getattr(machine, "current_state")
    assert getattr(state, "id", None) == machine.completed.id  # type: ignore[attr-defined]


def test_repair_machine_retries_until_exhaustion():
    attempts: list[int] = []

    def run_policy_pipeline(pipeline, repo_root, repair_obs=None):
        attempts.append(1)
        result = StepResult(name="step", success=False, exit_code=1, stdout="", stderr="boom")
        return [result], False

    def finalize_failed_repair(*, pipeline_results, classified, **kwargs):
        return FailedRepairDetails(
            failure_context="ctx",
            policy_code="policy",
            failed_step=pipeline_results[0].name,
            failed_exit_code=pipeline_results[0].exit_code,
            failed_summary="boom",
            error="pipeline_failed",
        )

    deps = _make_deps(
        run_policy_pipeline=run_policy_pipeline,
        finalize_failed_repair=finalize_failed_repair,
        classify_results=lambda results: PipelineClassification(test_failures="boom"),
    )
    repair_ctx = _make_repair_ctx(deps, max_repair_attempts=2)
    repair_model = RepairMachineModel(context=repair_ctx, deps=deps)
    machine = RepairStateMachine(repair_model)
    outcome = machine.run()
    assert outcome.success is False
    assert outcome.attempts_used == 2
    assert outcome.pipeline_feedback and outcome.pipeline_feedback.test_failures == "boom"
    state = getattr(machine, "current_state")
    assert getattr(state, "id", None) == machine.repair_exhausted.id  # type: ignore[attr-defined]


def test_repair_succeeds_on_second_attempt():
    pipeline_calls: list[int] = []

    def run_policy_pipeline(pipeline, repo_root, repair_obs=None):
        pipeline_calls.append(1)
        if len(pipeline_calls) == 1:
            result = StepResult(name="ruff", success=False, exit_code=1, stdout="", stderr="lint error")
            return [result], False
        result = StepResult(name="ruff", success=True, exit_code=0, stdout="", stderr="")
        return [result], True

    deps = _make_deps(
        run_policy_pipeline=run_policy_pipeline,
        classify_results=lambda results: PipelineClassification(lint_errors="bad"),
        finalize_failed_repair=lambda *, pipeline_results, classified, **kwargs: FailedRepairDetails(
            failure_context="ctx",
            policy_code="policy",
            failed_step=pipeline_results[0].name,
            failed_exit_code=pipeline_results[0].exit_code,
            failed_summary="lint error",
            error="pipeline_failed",
        ),
        finalize_successful_policy=lambda **kwargs: (
            "policy",
            AcceptedPolicyMeta(
                accepted_policy_source="post_pipeline_disk",
                accepted_policy_changed_by_pipeline=False,
                repair_attempts=1,
                max_policy_repairs=3,
            ),
        ),
    )
    repair_ctx = _make_repair_ctx(deps, max_repair_attempts=3)
    repair_model = RepairMachineModel(context=repair_ctx, deps=deps)
    machine = RepairStateMachine(repair_model)
    outcome = machine.run()
    assert outcome.success is True
    assert outcome.attempts_used == 2
    assert outcome.repair_attempts_reported == 1


def test_pipeline_mutation_returns_on_disk_policy():
    written_code: list[str] = []
    disk_policy = ["initial"]

    def write_policy(code):
        written_code.append(code)
        disk_policy[0] = code

    def run_policy_pipeline(pipeline, repo_root, repair_obs=None):
        disk_policy[0] = "formatted_policy"
        result = StepResult(name="ruff_fix", success=True, exit_code=0, stdout="", stderr="")
        return [result], True

    deps = _make_deps(
        write_policy=write_policy,
        read_policy=lambda: disk_policy[0],
        run_policy_pipeline=run_policy_pipeline,
        finalize_successful_policy=lambda **kwargs: (
            disk_policy[0],
            AcceptedPolicyMeta(
                accepted_policy_source="post_pipeline_disk",
                accepted_policy_changed_by_pipeline=disk_policy[0] != kwargs.get("new_code", ""),
                repair_attempts=0,
                max_policy_repairs=3,
            ),
        ),
    )
    repair_ctx = _make_repair_ctx(deps)
    repair_model = RepairMachineModel(context=repair_ctx, deps=deps)
    machine = RepairStateMachine(repair_model)
    outcome = machine.run()
    assert outcome.success is True
    assert outcome.policy == "formatted_policy"
    assert outcome.accepted_policy_source == "post_pipeline_disk"
    assert outcome.accepted_policy_changed_by_pipeline is True


def test_warning_only_steps_do_not_fail():
    def run_policy_pipeline(pipeline, repo_root, repair_obs=None):
        results = [
            StepResult(name="basedpyright", success=False, stdout="warning only", stderr="reportUnusedParameter", exit_code=0),
            StepResult(name="pytest_iterative_context", success=False, stdout="warning", stderr="", exit_code=0),
            StepResult(name="ruff_fix", success=False, stdout="notice", stderr="", exit_code=0),
        ]
        pipeline_results = results
        return pipeline_results, True

    deps = _make_deps(
        run_policy_pipeline=run_policy_pipeline,
        finalize_successful_policy=lambda **kwargs: (
            "policy",
            AcceptedPolicyMeta(
                accepted_policy_source="post_pipeline_disk",
                accepted_policy_changed_by_pipeline=False,
                repair_attempts=0,
                max_policy_repairs=3,
            ),
        ),
    )
    repair_ctx = _make_repair_ctx(deps)
    repair_model = RepairMachineModel(context=repair_ctx, deps=deps)
    machine = RepairStateMachine(repair_model)
    outcome = machine.run()
    assert outcome.success is True
    assert outcome.attempts_used == 1
    assert outcome.repair_attempts_reported == 0
    assert not outcome.pipeline_feedback
    assert outcome.failed_step is None


def test_nonzero_exit_still_fails():
    def run_policy_pipeline(pipeline, repo_root, repair_obs=None):
        result = StepResult(name="pytest_iterative_context", success=False, stdout="", stderr="test failed", exit_code=1)
        return [result], False

    def finalize_failed_repair(*, pipeline_results, classified, **kwargs):
        return FailedRepairDetails(
            failure_context="ctx",
            policy_code="policy",
            failed_step=pipeline_results[0].name,
            failed_exit_code=pipeline_results[0].exit_code,
            failed_summary="test failed",
            error="pipeline_failed",
        )

    deps = _make_deps(
        run_policy_pipeline=run_policy_pipeline,
        classify_results=lambda results: PipelineClassification(test_failures="boom"),
        finalize_failed_repair=finalize_failed_repair,
    )
    repair_ctx = _make_repair_ctx(deps, max_repair_attempts=2)
    repair_model = RepairMachineModel(context=repair_ctx, deps=deps)
    machine = RepairStateMachine(repair_model)
    outcome = machine.run()
    assert outcome.success is False
    assert outcome.attempts_used == 2
    assert outcome.pipeline_feedback and outcome.pipeline_feedback.test_failures == "boom"


def test_failed_candidate_not_marked_accepted():
    def run_policy_pipeline(pipeline, repo_root, repair_obs=None):
        result = StepResult(name="ruff_fix", success=False, stdout="", stderr="lint", exit_code=1)
        return [result], False

    def finalize_failed_repair(*, pipeline_results, classified, **kwargs):
        return FailedRepairDetails(
            failure_context="ctx",
            policy_code="policy",
            failed_step=pipeline_results[0].name,
            failed_exit_code=pipeline_results[0].exit_code,
            failed_summary="lint",
            error="pipeline_failed",
        )

    deps = _make_deps(
        run_policy_pipeline=run_policy_pipeline,
        classify_results=lambda results: PipelineClassification(lint_errors="missing"),
        finalize_failed_repair=finalize_failed_repair,
    )
    repair_ctx = _make_repair_ctx(deps, max_repair_attempts=2)
    repair_model = RepairMachineModel(context=repair_ctx, deps=deps)
    machine = RepairStateMachine(repair_model)
    outcome = machine.run()
    assert outcome.success is False
    assert outcome.accepted_policy_source is None
    assert outcome.accepted_policy_changed_by_pipeline is None


def test_score_metrics_propagation_into_iteration_record():
    deps = _make_deps(
        evaluate_policy_on_item=lambda ic_task, baseline, iteration_span=None, iteration_index=None: EvaluationResult(
            metrics={"score": 0.75, "coverage_delta": 0.1, "tool_error_rate": 0.05},
            ic_result={}, jc_result={}, jc_metrics={}, comparison_summary=None, policy_code="policy"
        ),
    )
    model = OptimizationMachineModel(
        context=LoopContext(task={"symbol": "s"}, iterations=1, baseline_snapshot=None, run_trace=None),
        deps=deps,
        max_policy_repairs=1,
    )
    machine = OptimizationStateMachine(model)
    history = machine.run()
    assert history
    record = history[0]
    assert record.metrics["score"] == 0.75
    assert record.metrics["coverage_delta"] == 0.1
    assert record.metrics["tool_error_rate"] == 0.05


def test_post_pipeline_metadata_propagates_into_accepted_policy():
    deps = _make_deps(
        finalize_successful_policy=lambda **kwargs: (
            "formatted_policy",
            AcceptedPolicyMeta(
                accepted_policy_source="post_pipeline_disk",
                accepted_policy_changed_by_pipeline=True,
                repair_attempts=0,
                max_policy_repairs=3,
            ),
        ),
    )
    model = OptimizationMachineModel(
        context=LoopContext(task={"symbol": "s"}, iterations=1, baseline_snapshot=None, run_trace=None),
        deps=deps,
        max_policy_repairs=3,
    )
    machine = OptimizationStateMachine(model)
    history = machine.run()
    assert history
    record = history[0]
    assert record.accepted_policy is not None
    assert record.accepted_policy.source == "post_pipeline_disk"
    assert record.accepted_policy.changed_by_pipeline is True


def test_repair_span_opens_before_writer_for_attempt_zero():
    span_log: list[tuple[str, dict[str, object]]] = []
    writer_parent_traces: list[object] = []

    class FakeSpan:
        def __init__(self, name: str, metadata: dict[str, object]):
            self.name = name
            self.metadata = metadata
            self.ended: list[dict[str, object]] = []

        def end(self, **kwargs: object) -> None:
            self.ended.append(kwargs)

    def tracking_start_span(parent=None, name=None, **kwargs):
        span = FakeSpan(name, kwargs.get("metadata", {}))
        span_log.append(("open", {"name": name, **kwargs.get("metadata", {})}))
        return span

    def tracking_attempt_policy_generation(**kwargs):
        writer_parent_traces.append(kwargs.get("parent_trace"))
        return ("policy", None)

    deps = _make_deps(
        start_observation=tracking_start_span,
        attempt_policy_generation=tracking_attempt_policy_generation,
    )
    repair_ctx = _make_repair_ctx(deps, max_repair_attempts=1)
    repair_model = RepairMachineModel(context=repair_ctx, deps=deps)
    machine = RepairStateMachine(repair_model)
    outcome = machine.run()
    assert outcome.success is True
    assert len(span_log) >= 1
    assert span_log[0][1]["attempt"] == 0
    assert writer_parent_traces[0] is not None


def test_repair_span_metadata_correct_across_retries():
    span_events: list[tuple[str, dict[str, object]]] = []
    pipeline_calls: list[int] = []

    class FakeSpan:
        def __init__(self, name: str, metadata: dict[str, object]):
            self.name = name
            self.metadata = metadata

        def end(self, **kwargs: object) -> None:
            meta = kwargs.get("metadata", {})
            meta_dict = {str(k): v for k, v in (meta.items() if isinstance(meta, Mapping) else [])}
            span_events.append(("close", meta_dict))

    def tracking_start_span(parent=None, name=None, **kwargs):
        meta = kwargs.get("metadata", {})
        meta_dict = {str(k): v for k, v in (meta.items() if isinstance(meta, Mapping) else [])}
        span = FakeSpan(name, meta_dict)
        span_events.append(("open", {"name": name, **meta_dict}))
        return span

    def run_policy_pipeline(pipeline, repo_root, repair_obs=None):
        pipeline_calls.append(1)
        if len(pipeline_calls) == 1:
            result = StepResult(name="ruff", success=False, exit_code=1, stdout="", stderr="err")
            return [result], False
        result = StepResult(name="ruff", success=True, exit_code=0, stdout="", stderr="")
        return [result], True

    deps = _make_deps(
        start_observation=tracking_start_span,
        run_policy_pipeline=run_policy_pipeline,
        classify_results=lambda results: PipelineClassification(lint_errors="bad"),
        finalize_failed_repair=lambda *, pipeline_results, classified, **kwargs: FailedRepairDetails(
            failure_context="ctx",
            policy_code="policy",
            failed_step=pipeline_results[0].name,
            failed_exit_code=pipeline_results[0].exit_code,
            failed_summary="err",
            error="pipeline_failed",
        ),
        finalize_successful_policy=lambda **kwargs: (
            "policy",
            AcceptedPolicyMeta(
                accepted_policy_source="post_pipeline_disk",
                accepted_policy_changed_by_pipeline=False,
                repair_attempts=1,
                max_policy_repairs=3,
            ),
        ),
    )
    repair_ctx = _make_repair_ctx(deps, max_repair_attempts=3)
    repair_model = RepairMachineModel(context=repair_ctx, deps=deps)
    machine = RepairStateMachine(repair_model)
    outcome = machine.run()
    assert outcome.success is True
    assert outcome.attempts_used == 2
    open_events = [e for e in span_events if e[0] == "open"]
    close_events = [e for e in span_events if e[0] == "close"]
    assert len(open_events) >= 2
    assert open_events[0][1]["attempt"] == 0
    assert open_events[1][1]["attempt"] == 1
    assert len(close_events) >= 2
    assert close_events[0][1].get("status") == "retrying"
    assert close_events[1][1].get("status") == "passed"
