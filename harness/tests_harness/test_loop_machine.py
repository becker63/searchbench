from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from harness.loop_machine import OptimizationStateMachine, RepairStateMachine
from harness.loop_types import (
    FeedbackPackage,
    EvaluationResult,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
    PipelineStepResult,
)
from harness.pipeline.types import StepResult


def test_optimization_machine_success_path():
    deps = LoopDependencies(
        prepare_iteration_tasks=lambda task, trace: PreparedTasks(
            base_task=task, resolved_repo_path=None, jc_repo_id=None
        ),
        evaluate_policy_on_item=lambda ic_task, baseline, iteration_span=None, iteration_index=None: EvaluationResult(
            metrics={"score": 1.0}, ic_result={}, jc_result={}, jc_metrics={}, comparison_summary=None, policy_code="policy"
        ),
        build_iteration_feedback=lambda evaluation, prev_score, prev_classified, repo_root: FeedbackPackage(
            tests="", scoring_context="", comparison_summary=None, feedback={"score": 1.0}, feedback_str="", guidance_hint="hint", diff_str="", diff_hint=""
        ),
        attempt_policy_generation=lambda **kwargs: ("policy", None),
        run_policy_pipeline=lambda pipeline, repo_root, repair_obs=None: (
            [PipelineStepResult(name="step", success=True, exit_code=0, stdout="", stderr="")],
            [StepResult(name="step", success=True, exit_code=0, stdout="", stderr="")],
            True,
        ),
        finalize_successful_policy=lambda **kwargs: (
            "policy",
            {
                "repair_attempts": 0,
                "max_policy_repairs": 1,
                "accepted_policy_source": "post_pipeline_disk",
                "accepted_policy_changed_by_pipeline": False,
            },
        ),
        finalize_failed_repair=lambda **kwargs: ("", "policy"),
        classify_results=lambda results: {},
        format_pipeline_failure_context=lambda *a, **k: "",
        read_policy=lambda: "policy",
        write_policy=lambda code: None,
        get_writer_model=lambda: "model",
        record_score=lambda *a, **k: None,
        start_span=lambda *a, **k: None,
        find_repo_root=lambda: Path("."),
        default_pipeline=lambda: object(),
    )
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

    def attempt_policy_generation(**kwargs):
        return "candidate", None

    def run_policy_pipeline(pipeline, repo_root, repair_obs=None):
        attempts.append(1)
        result = StepResult(name="step", success=False, exit_code=1, stdout="", stderr="boom")
        return [PipelineStepResult.from_mapping(result)], [result], False

    def finalize_failed_repair(*, metadata: dict[str, object], pipeline_results: Sequence[Mapping[str, object]], classified: Mapping[str, object], **kwargs):
        metadata["pipeline_feedback"] = classified
        metadata["failed_step"] = pipeline_results[0].get("name")
        metadata["failed_exit_code"] = pipeline_results[0].get("exit_code")
        metadata["failed_summary"] = "boom"
        return ("ctx", "policy")

    deps = LoopDependencies(
        prepare_iteration_tasks=lambda *a, **k: PreparedTasks(
            base_task={}, resolved_repo_path=None, jc_repo_id=None
        ),
        evaluate_policy_on_item=lambda *a, **k: evaluation,
        build_iteration_feedback=lambda *a, **k: feedback,
        attempt_policy_generation=attempt_policy_generation,
        run_policy_pipeline=run_policy_pipeline,
        finalize_successful_policy=lambda **kwargs: ("policy", {}),
        finalize_failed_repair=finalize_failed_repair,
        classify_results=lambda results: {"test_failures": "boom"},
        format_pipeline_failure_context=lambda *a, **k: "",
        read_policy=lambda: "policy",
        write_policy=lambda code: None,
        get_writer_model=lambda: "model",
        record_score=lambda *a, **k: None,
        start_span=lambda *a, **k: None,
        find_repo_root=lambda: Path("."),
        default_pipeline=lambda: object(),
    )
    evaluation = EvaluationResult(metrics={}, ic_result={}, jc_result={}, jc_metrics={}, comparison_summary=None, policy_code="policy")
    feedback = FeedbackPackage(tests="", scoring_context="", comparison_summary=None, feedback={}, feedback_str="", guidance_hint="", diff_str="", diff_hint="")
    repair_ctx = RepairContext(
        repo_root=Path("."),
        evaluation=evaluation,
        feedback=feedback,
        max_repair_attempts=2,
        parent_trace=None,
        writer_model="model",
    )
    repair_model = RepairMachineModel(context=repair_ctx, deps=deps)
    machine = RepairStateMachine(repair_model)
    outcome = machine.run()
    assert outcome.success is False
    assert outcome.attempts_used == 2
    assert outcome.metadata.get("pipeline_feedback") == {"test_failures": "boom"}
    state = getattr(machine, "current_state")
    assert getattr(state, "id", None) == machine.repair_exhausted.id  # type: ignore[attr-defined]
