from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Tuple

from harness.localization.errors import (
    LocalizationEvaluationError,
    LocalizationFailureCategory,
)
from harness.localization.runtime.records import build_localization_score_eval_record
from harness.localization.scoring import build_localization_score_context
from harness.localization.materialization.worktree import (
    RepoMaterializationRequest,
    RepoMaterializationResult,
    WorktreeManager,
)
from harness.localization.models import (
    LCATask,
    LocalizationEvidence,
    LocalizationPrediction,
    LocalizationRunResult,
)
from harness.scoring import ScoreBundle, ScoreEngine, summarize_task_score
from harness.localization.telemetry import build_localization_telemetry
from harness.scoring.token_usage import (
    TokenUsageRecord,
    extract_token_usage_record,
)
from harness.telemetry.tracing import start_observation
from harness.telemetry.tracing.score_emitter import (
    emit_score_bundle,
    emit_score_for_handle,
    score_bundle_metadata,
)


def _safe_update_metadata(span: object | None, metadata: Mapping[str, object]) -> None:
    updater = getattr(span, "update", None)
    if callable(updater):
        try:
            updater(metadata=metadata)
        except Exception:
            pass


def _get_repo_checkout(
    task: LCATask, worktree_manager: WorktreeManager | None
) -> RepoMaterializationResult | None:
    if worktree_manager is None:
        return None
    request = RepoMaterializationRequest(
        dataset_name=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        repo_owner=task.identity.repo_owner,
        repo_name=task.identity.repo_name,
        base_sha=task.identity.base_sha,
    )
    return worktree_manager.get_or_create(request)


def _resolve_repo_path(
    task: LCATask, checkout_result: RepoMaterializationResult | None
) -> Path:
    repo_path: Path | None = (
        checkout_result.local_path if checkout_result else None
    )
    if repo_path is None and task.repo:
        repo_path = Path(task.repo).expanduser()
    if repo_path is None:
        raise LocalizationEvaluationError(
            LocalizationFailureCategory.MATERIALIZATION,
            "Localization task is missing a repo path; provide a WorktreeManager or task.repo",
            task_id=task.task_id,
        )
    if not repo_path.exists():
        raise LocalizationEvaluationError(
            LocalizationFailureCategory.MATERIALIZATION,
            f"Localization repo path does not exist: {repo_path}",
            task_id=task.task_id,
        )
    return repo_path


def _run_runner(
    task: LCATask,
    repo_path: Path,
    trace: object | None,
    runner: Callable[
        [LCATask, str, object | None], LocalizationRunResult
    ]
    | None,
) -> LocalizationRunResult:
    runner_fn = runner or _default_runner
    try:
        runner_result = runner_fn(task, str(repo_path), trace)
    except LocalizationEvaluationError:
        raise
    except Exception as exc:
        raise LocalizationEvaluationError(
            LocalizationFailureCategory.RUNNER, str(exc), task_id=task.task_id
        ) from exc
    if not runner_result.prediction.predicted_files:
        raise LocalizationEvaluationError(
            LocalizationFailureCategory.RUNNER,
            "Localization runner produced no predicted_files",
            task_id=task.task_id,
        )
    return runner_result


def _score_prefix_from_runner_result(runner_result: LocalizationRunResult | None) -> str:
    if runner_result is None:
        return "localization"
    backend = runner_result.backend
    if backend in {"ic", "iterative_context"}:
        return "ic"
    if backend in {"jc", "jcodemunch"}:
        return "jc"
    source = runner_result.source
    if source in {"ic", "iterative_context"}:
        return "ic"
    if source in {"jc", "jcodemunch"}:
        return "jc"
    return "localization"


def _default_runner(
    lca_task: LCATask, repo_path: str, parent: object | None
) -> LocalizationRunResult:
    from harness.agents.localizer import run_ic_iteration as run_ic_iteration_fn
    result = run_ic_iteration_fn(lca_task.with_repo(repo_path), score_fn=None, steps=5, parent_trace=parent)
    if not isinstance(result, LocalizationRunResult):
        raise RuntimeError("Localization runner returned an invalid result")
    predicted = result.prediction.predicted_files
    if not isinstance(predicted, list):
        raise RuntimeError("Localization runner produced no predicted_files")
    predictions = [p for p in predicted if isinstance(p, str) and p.strip()]
    if not predictions:
        source = result.source
        if source == "runner_error":
            raw = result.raw
            error = raw.get("error") if isinstance(raw, Mapping) else None
            detail = f": {error}" if isinstance(error, str) and error else ""
            raise LocalizationEvaluationError(
                LocalizationFailureCategory.RUNNER,
                f"Localization runner failed before producing observations{detail}",
                task_id=lca_task.task_id,
                observability_summary=result.observability_summary,
            )
        detail = (
            " after finalization and fallback recovery"
            if source == "fallback_empty"
            else ""
        )
        raise LocalizationEvaluationError(
            LocalizationFailureCategory.RUNNER,
            f"Localization runner produced an empty predicted_files list{detail}",
            task_id=lca_task.task_id,
            observability_summary=result.observability_summary,
        )
    return result


def _score_task(
    task: LCATask, prediction: LocalizationPrediction, repo_path: Path, token_usage: TokenUsageRecord | None = None
) -> ScoreBundle:
    try:
        anchor_text = "\n".join([task.context.issue_title or "", task.context.issue_body or ""]).strip()
        score_context = build_localization_score_context(
            prediction=prediction,
            gold=task.gold,
            repo_path=repo_path,
            anchor_text=anchor_text,
            token_usage=token_usage,
        )
        score_bundle = ScoreEngine().evaluate(score_context)
        eval_record = build_localization_score_eval_record(
            task.identity,
            prediction,
            task.gold,
            score_context=score_context,
            score_bundle=score_bundle,
            evidence=task.evidence,
            repo_path=repo_path,
        )
        return eval_record.score_bundle
    except LocalizationEvaluationError:
        raise
    except Exception as exc:
        raise LocalizationEvaluationError(
            LocalizationFailureCategory.SCORING, str(exc), task_id=task.task_id
        ) from exc


def _emit_telemetry(
    task: LCATask,
    score_bundle: ScoreBundle,
    evidence: LocalizationEvidence | None,
    materialization: RepoMaterializationResult | None,
    dataset_provenance: str | None,
    trace: object | None,
    predicted_files: list[str],
    changed_files: list[str],
    score_prefix: str = "localization",
) -> None:
    task_score_summary = summarize_task_score(task.task_id, score_bundle)
    telemetry = build_localization_telemetry(
        task.identity,
        score_summary=task_score_summary,
        changed_files_count=len(task.gold.normalized_changed_files()),
        dataset_provenance=dataset_provenance,
        repo_language=task.context.repo_language,
        repo_license=task.context.repo_license,
        evidence=evidence,
        materialization_events=materialization.events if materialization else None,
    )
    emit_score_bundle(
        trace,
        score_bundle,
        prefix=score_prefix,
        metadata_key="score_bundle",
        emit_fn=emit_score_for_handle,
    )
    if trace:
        try:
            telemetry_payload = telemetry.model_dump(exclude_none=True)
            metadata_payload: dict[str, object] = {
                "telemetry": telemetry_payload,
                "score_summary": task_score_summary.model_dump(mode="json"),
                "predicted_files": list(predicted_files),
                "changed_files": list(changed_files),
            }
            metadata_payload.update(
                score_bundle_metadata(score_bundle, metadata_key="score_bundle")
            )
            _safe_update_metadata(trace, metadata_payload)
        except Exception:
            pass


def run_localization_task(
    task: LCATask,
    dataset_provenance: str | None = None,
    worktree_manager: WorktreeManager | None = None,
    parent_trace: object | None = None,
    runner: Callable[
        [LCATask, str, object | None], LocalizationRunResult
    ]
    | None = None,
) -> Tuple[
    LocalizationPrediction,
    ScoreBundle,
    LocalizationEvidence | None,
    RepoMaterializationResult | None,
    TokenUsageRecord,
    LocalizationRunResult,
]:
    """
    Execute a single localization task end to end (leaf helper):
    - resolve repo@sha checkout through WorktreeManager when provided
    - run the localization agent/backend to produce predicted_files
    - score using static-graph score bundles
    - emit task-level telemetry
    Returns prediction, score bundle, optional evidence, checkout details, and token usage.
    """

    with start_observation(
        name="localization_task",
        parent=parent_trace,
        metadata={
            "identity": task.task_id,
            "dataset": task.identity.dataset_name,
            "dataset_config": task.identity.dataset_config,
            "dataset_split": task.identity.dataset_split,
            "repo_owner": task.identity.repo_owner,
            "repo_name": task.identity.repo_name,
            "base_sha": task.identity.base_sha,
        },
    ) as trace:
        try:
            checkout_result = _get_repo_checkout(task, worktree_manager)
        except Exception as exc:
            raise LocalizationEvaluationError(
                LocalizationFailureCategory.MATERIALIZATION,
                str(exc),
                task_id=task.task_id,
            ) from exc
        repo_path = _resolve_repo_path(task, checkout_result)
        runner_result = _run_runner(task, repo_path, trace, runner)
        prediction = runner_result.prediction
        usage = extract_token_usage_record(
            runner_result.model_dump(),
            source="runner",
        )
        score_bundle = _score_task(task, prediction, repo_path, usage)
        if runner_result.observability_summary is not None:
            predicted_set = set(prediction.normalized_predicted_files())
            gold_set = set(task.gold.normalized_changed_files())
            overlap = len(predicted_set & gold_set)
            precision = overlap / len(predicted_set) if predicted_set else 0.0
            recall = overlap / len(gold_set) if gold_set else 0.0
            runner_result.observability_summary.evaluation.update(
                {
                    "gold_overlap_count": float(overlap),
                    "file_precision": precision,
                    "file_recall": recall,
                }
            )
        _emit_telemetry(
            task=task,
            score_bundle=score_bundle,
            evidence=task.evidence,
            materialization=checkout_result,
            dataset_provenance=dataset_provenance,
            trace=trace,
            predicted_files=prediction.normalized_predicted_files(),
            changed_files=task.gold.normalized_changed_files(),
            score_prefix=_score_prefix_from_runner_result(runner_result),
        )
        if runner_result.observability_summary is not None:
            try:
                _safe_update_metadata(
                    trace,
                    {
                        "task_summary": runner_result.observability_summary.model_dump(
                            mode="json"
                        )
                    },
                )
            except Exception:
                pass
        return prediction, score_bundle, task.evidence, checkout_result, usage, runner_result
