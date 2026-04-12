from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Tuple

from harness.localization.errors import (
    LocalizationEvaluationError,
    LocalizationFailureCategory,
)
from harness.localization.runtime.records import build_file_localization_eval_record
from harness.localization.materialization.materialize import (
    RepoMaterializationRequest,
    RepoMaterializationResult,
    RepoMaterializer,
)
from harness.localization.models import (
    LCAPrediction,
    LCATask,
    LocalizationEvidence,
    LocalizationMetrics,
)
from harness.localization.telemetry import build_localization_telemetry
from harness.localization.token_usage import (
    TokenUsageRecord,
    extract_token_usage_record,
)
from harness.telemetry.tracing import start_observation
from harness.telemetry.tracing.score_emitter import emit_score_for_handle


def _safe_update_metadata(span: object | None, metadata: Mapping[str, object]) -> None:
    updater = getattr(span, "update", None)
    if callable(updater):
        try:
            updater(metadata=metadata)
        except Exception:
            pass


def _materialize_repo(
    task: LCATask, materializer: RepoMaterializer | None
) -> RepoMaterializationResult | None:
    if materializer is None:
        return None
    request = RepoMaterializationRequest(
        dataset_name=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        repo_owner=task.identity.repo_owner,
        repo_name=task.identity.repo_name,
        base_sha=task.identity.base_sha,
    )
    return materializer.materialize(request)


def _resolve_repo_path(
    task: LCATask, materialization_result: RepoMaterializationResult | None
) -> Path:
    repo_path: Path | None = (
        materialization_result.local_path if materialization_result else None
    )
    if repo_path is None and task.repo:
        repo_path = Path(task.repo).expanduser()
    if repo_path is None:
        raise LocalizationEvaluationError(
            LocalizationFailureCategory.MATERIALIZATION,
            "Localization task is missing a repo path; provide a materializer or task.repo",
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
        [LCATask, str, object | None], tuple[list[str], Mapping[str, object] | None]
    ]
    | None,
) -> tuple[list[str], Mapping[str, object] | None]:
    runner_fn = runner or _default_runner
    try:
        predicted_files, runner_result = runner_fn(task, str(repo_path), trace)
    except LocalizationEvaluationError:
        raise
    except Exception as exc:
        raise LocalizationEvaluationError(
            LocalizationFailureCategory.RUNNER, str(exc), task_id=task.task_id
        ) from exc
    if not predicted_files:
        raise LocalizationEvaluationError(
            LocalizationFailureCategory.RUNNER,
            "Localization runner produced no predicted_files",
            task_id=task.task_id,
        )
    result_map = runner_result if isinstance(runner_result, Mapping) else None
    return predicted_files, result_map


def _default_runner(
    lca_task: LCATask, repo_path: str, parent: object | None
) -> tuple[list[str], Mapping[str, object] | None]:
    from harness.agents.localizer import run_ic_iteration as run_ic_iteration_fn
    result = run_ic_iteration_fn(
        {
            "identity": lca_task.identity.model_dump(),
            "context": lca_task.context.model_dump(),
            "repo": repo_path,
        },
        score_fn=None,
        steps=5,
        parent_trace=parent,
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("Localization runner returned an invalid result")
    predicted = result.get("predicted_files")
    if not isinstance(predicted, list):
        raise RuntimeError("Localization runner produced no predicted_files")
    predictions = [p for p in predicted if isinstance(p, str) and p.strip()]
    if not predictions:
        raise RuntimeError("Localization runner produced an empty predicted_files list")
    return predictions, result


def _score_task(
    task: LCATask, prediction: LCAPrediction, repo_path: Path
) -> LocalizationMetrics:
    try:
        eval_record = build_file_localization_eval_record(
            task.identity,
            prediction,
            task.gold,
            evidence=task.evidence,
            repo_path=repo_path,
        )
        return eval_record.metrics
    except LocalizationEvaluationError:
        raise
    except Exception as exc:
        raise LocalizationEvaluationError(
            LocalizationFailureCategory.SCORING, str(exc), task_id=task.task_id
        ) from exc


def _emit_telemetry(
    task: LCATask,
    metrics: LocalizationMetrics,
    evidence: LocalizationEvidence | None,
    materialization: RepoMaterializationResult | None,
    dataset_source: str | None,
    trace: object | None,
    predicted_files: list[str],
    changed_files: list[str],
) -> None:
    telemetry = build_localization_telemetry(
        task.identity,
        metrics=metrics,
        changed_files_count=len(task.gold.normalized_changed_files()),
        dataset_source=dataset_source,
        repo_language=task.context.repo_language,
        repo_license=task.context.repo_license,
        evidence=evidence,
        materialization_events=materialization.events if materialization else None,
    )
    for metric_name, metric_value in metrics.model_dump().items():
        if metric_value is None:
            continue
        emit_score_for_handle(
            trace,
            name=f"localization.{metric_name}",
            value=float(metric_value),
            data_type="NUMERIC",
            score_id=f"{getattr(trace, 'id', None)}-localization.{metric_name}"
            if getattr(trace, "id", None)
            else None,
        )
    if trace:
        try:
            metadata_payload = {
                "telemetry": telemetry.model_dump(exclude_none=True),
                "predicted_files": list(predicted_files),
                "changed_files": list(changed_files),
            }
            _safe_update_metadata(trace, metadata_payload)
        except Exception:
            pass


def run_localization_task(
    task: LCATask,
    dataset_source: str | None = None,
    materializer: RepoMaterializer | None = None,
    parent_trace: object | None = None,
    runner: Callable[
        [LCATask, str, object | None], tuple[list[str], Mapping[str, object] | None]
    ]
    | None = None,
) -> Tuple[
    LCAPrediction,
    LocalizationMetrics,
    LocalizationEvidence | None,
    RepoMaterializationResult | None,
    TokenUsageRecord,
]:
    """
    Execute a single localization task end to end (leaf helper):
    - materialize repo at base_sha (if a materializer is provided)
    - run the localization agent/backend to produce predicted_files
    - score using file-set localization metrics
    - emit task-level telemetry
    Returns prediction, metrics, optional evidence, and materialization details.
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
            materialization_result = _materialize_repo(task, materializer)
        except Exception as exc:
            raise LocalizationEvaluationError(
                LocalizationFailureCategory.MATERIALIZATION,
                str(exc),
                task_id=task.task_id,
            ) from exc
        repo_path = _resolve_repo_path(task, materialization_result)
        predicted_files, runner_result = _run_runner(task, repo_path, trace, runner)
        prediction = LCAPrediction(predicted_files=predicted_files)
        metrics = _score_task(task, prediction, repo_path)
        usage = extract_token_usage_record(
            runner_result if isinstance(runner_result, Mapping) else None,
            source="runner",
        )
        _emit_telemetry(
            task=task,
            metrics=metrics,
            evidence=task.evidence,
            materialization=materialization_result,
            dataset_source=dataset_source,
            trace=trace,
            predicted_files=prediction.normalized_predicted_files(),
            changed_files=task.gold.normalized_changed_files(),
        )
        return prediction, metrics, task.evidence, materialization_result, usage
