from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping

# Ensure the repository root (parent of the harness package) is on sys.path when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.loop_machine import OptimizationStateMachine, RepairStateMachine
from harness.loop_types import (
    AcceptedPolicyMeta,
    FailedRepairDetails,
    EvaluationResult,
    FeedbackPackage,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
)
from harness.pipeline.types import PipelineClassification, StepResult


def _stub_deps() -> LoopDependencies:
    """Minimal dependency set for chart rendering (no side effects)."""

    def _prepare(task: Mapping[str, object], trace: object | None = None) -> PreparedTasks:
        return PreparedTasks(base_task=dict(task), resolved_repo_path=None, jc_repo_id=None)

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

    def _evaluate_policy_on_item(*args: object, **kwargs: object) -> EvaluationResult:
        return dummy_eval

    def _build_iteration_feedback(*args: object, **kwargs: object) -> FeedbackPackage:
        return dummy_feedback

    def _attempt_policy_generation(**kwargs: object) -> tuple[str | None, str | None]:
        return ("policy", None)

    def _run_policy_pipeline(pipeline: object, repo_root: Path, repair_obs: object | None = None) -> tuple[list[StepResult], bool]:
        return ([StepResult(name="step", success=True, exit_code=0, stdout="", stderr="")], True)

    def _finalize_successful_policy(**kwargs: object) -> tuple[str, AcceptedPolicyMeta]:
        return (
            "policy",
            AcceptedPolicyMeta(
                accepted_policy_source=None,
                accepted_policy_changed_by_pipeline=None,
                repair_attempts=0,
                max_policy_repairs=1,
            ),
        )

    def _finalize_failed_repair(**kwargs: object) -> FailedRepairDetails:
        return FailedRepairDetails(
            failure_context="ctx",
            policy_code="policy",
            failed_step=None,
            failed_exit_code=None,
            failed_summary=None,
            error="pipeline_failed",
        )

    def _format_pf_ctx(
        pipeline_results: list[StepResult],
        classified: PipelineClassification | None,
        writer_error: str | None,
        model_name: str | None,
        attempt: int,
    ) -> str:
        return ""

    def _record_score(handle: object | None, name: str, value: float, metadata: dict[str, object] | None = None) -> None:
        return None

    return LoopDependencies(
        prepare_iteration_tasks=_prepare,
        evaluate_policy_on_item=_evaluate_policy_on_item,
        build_iteration_feedback=_build_iteration_feedback,
        attempt_policy_generation=_attempt_policy_generation,
        run_policy_pipeline=_run_policy_pipeline,
        finalize_successful_policy=_finalize_successful_policy,
        finalize_failed_repair=_finalize_failed_repair,
        classify_results=lambda results: PipelineClassification(),
        format_pipeline_failure_context=_format_pf_ctx,
        read_policy=lambda: "policy",
        write_policy=lambda code: None,
        get_writer_model=lambda: "model",
        record_score=_record_score,
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


def _render(machine: object, fmt: str) -> str:
    # python-statemachine exposes a private graph builder; use it to support multiple formats.
    graph = machine._graph()  # type: ignore[attr-defined]
    if fmt == "mermaid":
        try:
            return f"{machine:mermaid}"
        except Exception:
            return graph.to_string()  # type: ignore[attr-defined]
    if fmt == "dot":
        try:
            return f"{machine:dot}"
        except Exception:
            return graph.to_string()  # type: ignore[attr-defined]
    if fmt == "md":
        try:
            return f"{machine:md}"
        except Exception:
            return graph.to_string()  # type: ignore[attr-defined]
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
        for idx, (name, chart) in enumerate(targets):
            if machine == "both" and idx > 0:
                path = output.with_name(f"{output.stem}_{name}{output.suffix}")
            else:
                path = output
            graph = chart._graph()  # type: ignore[attr-defined]
            try:
                graph.write_png(str(path))
            except FileNotFoundError as exc:
                raise RuntimeError("Graphviz 'dot' executable is required for PNG output") from exc
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
