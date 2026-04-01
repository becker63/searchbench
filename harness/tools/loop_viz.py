from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

from harness.loop_machine import OptimizationStateMachine, RepairStateMachine
from harness.loop_types import (
    EvaluationResult,
    FeedbackPackage,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
)
from harness.pipeline.types import StepResult


def _stub_deps() -> LoopDependencies:
    """Minimal dependency set for chart rendering (no side effects)."""

    def _prepare(task: Mapping[str, object], trace: object | None = None) -> PreparedTasks:
        return PreparedTasks(base_task=task, resolved_repo_path=None, jc_repo_id=None)

    dummy_eval = EvaluationResult(metrics={}, ic_result={}, jc_result={}, jc_metrics={}, comparison_summary=None, policy_code="")
    dummy_feedback = FeedbackPackage(
        tests="",
        scoring_context="",
        comparison_summary=None,
        feedback={},
        feedback_str="",
        guidance_hint="",
        diff_str="",
        diff_hint="",
    )

    return LoopDependencies(
        prepare_iteration_tasks=_prepare,
        evaluate_policy_on_item=lambda *a, **k: dummy_eval,
        build_iteration_feedback=lambda *a, **k: dummy_feedback,
        attempt_policy_generation=lambda **kwargs: ("policy", None),
        run_policy_pipeline=lambda pipeline, repo_root, repair_obs=None: (
            [StepResult(name="step", success=True, exit_code=0, stdout="", stderr="")],
            [StepResult(name="step", success=True, exit_code=0, stdout="", stderr="")],
            True,
        ),
        finalize_successful_policy=lambda **kwargs: ("policy", {}),
        finalize_failed_repair=lambda **kwargs: ("ctx", "policy"),
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


def _build_charts() -> tuple[RepairStateMachine, OptimizationStateMachine]:
    deps = _stub_deps()
    repair_ctx = RepairContext(
        repo_root=Path("."),
        evaluation=EvaluationResult(metrics={}, ic_result={}, jc_result={}, jc_metrics={}, comparison_summary=None, policy_code=""),
        feedback=FeedbackPackage(
            tests="", scoring_context="", comparison_summary=None, feedback={}, feedback_str="", guidance_hint="", diff_str="", diff_hint=""
        ),
        max_repair_attempts=1,
        parent_trace=None,
        writer_model="model",
    )
    repair_model = RepairMachineModel(context=repair_ctx, deps=deps)
    repair_chart = RepairStateMachine(repair_model)

    loop_ctx = LoopContext(task={}, iterations=1, baseline_snapshot=None, run_trace=None)
    opt_model = OptimizationMachineModel(context=loop_ctx, deps=deps, max_policy_repairs=1)
    opt_chart = OptimizationStateMachine(opt_model)
    return repair_chart, opt_chart


def _render(machine, fmt: str) -> str:
    if fmt == "mermaid":
        return f"{machine:mermaid}"
    if fmt == "dot":
        return f"{machine:dot}"
    if fmt == "md":
        return f"{machine:md}"
    raise ValueError(f"Unsupported format: {fmt}")


def render_state_machines(fmt: str = "mermaid", machine: str = "both", output: Path | None = None) -> None:
    repair_chart, opt_chart = _build_charts()
    targets = []
    if machine in ("both", "repair"):
        targets.append(("RepairStateMachine", repair_chart))
    if machine in ("both", "optimization"):
        targets.append(("OptimizationStateMachine", opt_chart))

    if fmt == "png":
        if output is None:
            raise ValueError("PNG output requires --output path")
        for name, chart in targets:
            path = output if machine != "both" else output.with_name(f"{output.stem}_{name}{output.suffix}")
            graph = chart._graph()  # type: ignore[attr-defined]
            graph.write_png(str(path))
        return

    rendered = []
    for name, chart in targets:
        rendered.append(f"%% {name}\n{_render(chart, fmt)}")
    text = "\n\n".join(rendered)
    if output:
        output.write_text(text, encoding="utf-8")
    else:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render loop state machines (uses python-statemachine rendering).")
    parser.add_argument("--format", choices=["mermaid", "dot", "md", "png"], default="mermaid", help="Output format")
    parser.add_argument(
        "--machine",
        choices=["optimization", "repair", "both"],
        default="both",
        help="Which machine to render",
    )
    parser.add_argument("--output", type=Path, help="Output file path. Required for png, optional otherwise.")
    args = parser.parse_args()
    render_state_machines(fmt=args.format, machine=args.machine, output=args.output)


if __name__ == "__main__":
    main()
