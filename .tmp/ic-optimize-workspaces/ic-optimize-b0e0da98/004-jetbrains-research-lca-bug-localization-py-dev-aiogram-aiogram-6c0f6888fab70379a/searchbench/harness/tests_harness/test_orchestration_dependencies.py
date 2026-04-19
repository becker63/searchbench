from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.localization.runtime.evaluate import LocalizationEvaluationResult
from harness.orchestration.dependencies import RuntimeCollaborators, build_loop_dependencies
from harness.orchestration.types import (
    AcceptedPolicyMeta,
    EvaluationMetrics,
    EvaluationResult,
    FailedRepairDetails,
    FeedbackPackage,
    ICResult,
    JCResult,
)
from harness.pipeline.types import PipelineClassification, StepResult
from harness.prompts import WriterOptimizationBrief, build_writer_optimization_brief
from harness.utils.diff import DiffSummary
from harness.utils.type_loader import FrontierContext


class RecordingSpan:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def end(self, **kwargs: object) -> None:
        self.updates.append(dict(kwargs))

    def update(self, **kwargs: object) -> None:
        self.updates.append(dict(kwargs))


class EmptyPipeline:
    def run(
        self,
        repo_root: Path,
        observation: RecordingSpan | None = None,
        allow_no_parent: bool = False,
    ) -> list[StepResult]:
        return []


def _task() -> LCATask:
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    return LCATask(
        identity=identity,
        context=LCAContext(issue_title="bug", issue_body="details"),
        gold=LCAGold(changed_files=["a.py"]),
        repo="repo",
    )


def _evaluation() -> EvaluationResult:
    return EvaluationResult(
        metrics=EvaluationMetrics.model_validate({"score": 1.0}),
        ic_result=ICResult(),
        jc_result=JCResult(),
        jc_metrics=EvaluationMetrics(),
        comparison_summary=None,
        policy_code="policy",
    )


def _feedback() -> FeedbackPackage:
    frontier = FrontierContext(signature="", graph_models="", types="", examples="", notes="")
    return FeedbackPackage(
        tests="tests",
        frontier_context="frontier",
        frontier_context_details=frontier,
        optimization_brief=build_writer_optimization_brief(guidance=["hint"]),
    )


def test_build_loop_dependencies_wires_explicit_collaborators(tmp_path: Path) -> None:
    spans: list[RecordingSpan] = []
    indexed: list[str] = []
    emitted_scores: list[str] = []
    pipeline = EmptyPipeline()

    @contextmanager
    def start_observation(
        name: str,
        *,
        parent: object | None = None,
        as_type: str = "span",
        input: object | None = None,
        model: str | None = None,
        metadata: dict[str, object] | None = None,
        usage_details: object | None = None,
    ) -> Iterator[RecordingSpan]:
        span = RecordingSpan()
        spans.append(span)
        yield span

    def generate_policy(
        current_policy: str,
        tests: str,
        frontier_context: str,
        frontier_context_details: FrontierContext,
        optimization_brief: WriterOptimizationBrief,
        feedback: dict[str, object] | None = None,
        parent_trace: object | None = None,
        failure_context: str | None = None,
        repair_attempt: int = 0,
    ) -> str:
        return current_policy

    collaborators = RuntimeCollaborators(
        index_folder=lambda repo: indexed.append(repo) or {"repo": "indexed-repo"},
        evaluate_localization_batch=lambda tasks, **kwargs: LocalizationEvaluationResult(),
        load_tests=lambda repo_root: [],
        format_tests_for_prompt=lambda tests: "tests",
        build_frontier_context=lambda repo_root: FrontierContext(signature="", graph_models="", types="", examples="", notes=""),
        format_frontier_context=lambda ctx: "frontier",
        compute_diff=lambda prev, current, old, new: DiffSummary(0.0, set(), set(), False),
        format_diff=lambda diff: "diff",
        interpret_diff=lambda diff: "guidance",
        generate_policy=generate_policy,
        emit_score=lambda **kwargs: emitted_scores.append(str(kwargs["name"])),
        classify_results=lambda results: PipelineClassification(passed=[step.name for step in results]),
        read_policy=lambda: "policy",
        write_policy=lambda code: None,
        get_writer_model=lambda: "model",
        start_observation=start_observation,
        find_repo_root=lambda: tmp_path,
        default_pipeline=lambda: pipeline,
    )

    deps = build_loop_dependencies(
        collaborators,
        evaluate_policy_on_item=lambda task, baseline, span, index: _evaluation(),
        build_iteration_feedback=lambda evaluation, prev, classified, repo: _feedback(),
        attempt_policy_generation=lambda **kwargs: ("policy", None),
    )

    prepared = deps.prepare_iteration_tasks(_task(), None)
    assert prepared.ic_task().repo == "repo"
    assert prepared.jc_task().repo == "indexed-repo"
    assert indexed == ["repo"]
    assert spans and spans[0].updates == [{"metadata": {"repo": "indexed-repo"}}]

    assert deps.read_policy() == "policy"
    assert deps.find_repo_root() == tmp_path
    assert deps.default_pipeline() is pipeline
    assert deps.classify_results([StepResult(name="step", success=True, exit_code=0, stdout="", stderr="")]).passed == ["step"]
    assert deps.run_policy_pipeline(pipeline, tmp_path) == ([], True)
    assert emitted_scores == []
    assert deps.finalize_successful_policy(
        new_code="candidate",
        attempts_used=2,
        max_repair_attempts=3,
    ) == (
        "policy",
        AcceptedPolicyMeta(
            accepted_policy_source="post_pipeline_disk",
            accepted_policy_changed_by_pipeline=True,
            repair_attempts=1,
            max_policy_repairs=3,
        ),
    )
    assert isinstance(
        deps.finalize_failed_repair(
            pipeline_results=[StepResult(name="step", success=False, exit_code=1, stdout="", stderr="bad")],
            classified=PipelineClassification(type_errors="bad"),
            writer_model="model",
            attempts_used=1,
            writer_error=None,
        ),
        FailedRepairDetails,
    )


def test_refactored_orchestration_surface_has_no_cast_usage() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "orchestration" / "runtime.py").exists()
    paths = [
        root / "orchestration" / "dependencies.py",
    ]

    offenders = [
        str(path.relative_to(root))
        for path in paths
        if "cast(" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
