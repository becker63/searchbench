from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Tuple

from harness.localization.eval import build_file_localization_eval_record
from harness.localization.materializer import (
    RepoMaterializationRequest,
    RepoMaterializationResult,
    RepoMaterializer,
    describe_repo_mapping,
)
from harness.localization.models import (
    LCAGold,
    LCAPrediction,
    LCATask,
    LocalizationEvidence,
    LocalizationMetrics,
)
from harness.localization.telemetry import build_localization_telemetry
from harness.observability.langfuse import start_observation
from harness.observability.score_emitter import emit_score_for_handle
from harness.loop.runner_agent import run_ic_iteration


def _materialize_repo(task: LCATask, materializer: RepoMaterializer | None) -> RepoMaterializationResult | None:
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


def run_localization_task(
    task: LCATask,
    dataset_source: str | None = None,
    materializer: RepoMaterializer | None = None,
    parent_trace: object | None = None,
    runner: Callable[[LCATask, str, object | None], tuple[list[str], Mapping[str, object] | None]] | None = None,
) -> Tuple[LCAPrediction, LocalizationMetrics, LocalizationEvidence | None, RepoMaterializationResult | None]:
    """
    Execute a localization task end to end:
    - materialize repo at base_sha (if a materializer is provided)
    - run the localization agent/backend to produce predicted_files
    - score using file-set localization metrics
    - emit scores via Langfuse
    Returns prediction, metrics, optional evidence, and materialization details.
    """

    def _default_runner(lca_task: LCATask, repo_path: str, parent: object | None) -> tuple[list[str], Mapping[str, object] | None]:
        result = run_ic_iteration(
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
            # Surface materialization failures with category hint if available
            hint = getattr(exc, "category", None)
            raise RuntimeError(f"Materialization failed ({hint or 'materialization'}): {exc}") from exc
        repo_path: Path | None = materialization_result.local_path if materialization_result else None
        if repo_path is None and task.repo:
            repo_path = Path(task.repo).expanduser()
        if repo_path is not None and not repo_path.exists():
            raise RuntimeError(f"Localization repo path does not exist: {repo_path}")
        if repo_path is None:
            raise RuntimeError("Localization task is missing a repo path; provide a materializer or task.repo")
        runner_fn = runner or _default_runner
        predicted_files, _runner_result = runner_fn(task, str(repo_path), trace)
        if not predicted_files:
            raise RuntimeError("Localization runner produced no predicted_files")
        prediction = LCAPrediction(predicted_files=predicted_files)
        eval_record = build_file_localization_eval_record(
            task.identity, prediction, task.gold, evidence=task.evidence, repo_path=repo_path
        )
        telemetry = build_localization_telemetry(
            task.identity,
            metrics=eval_record.metrics,
            changed_files_count=len(task.gold.normalized_changed_files()),
            dataset_source=dataset_source,
            repo_language=task.context.repo_language,
            repo_license=task.context.repo_license,
            evidence=task.evidence,
            materialization_events=materialization_result.events if materialization_result else None,
        )
        dataset_run_id = getattr(parent_trace, "id", None)
        for metric_name, metric_value in eval_record.metrics.model_dump().items():
            if metric_value is None:
                continue
            emit_score_for_handle(
                trace,
                name=f"localization.{metric_name}",
                value=float(metric_value),
                data_type="NUMERIC",
                dataset_run_id=dataset_run_id,
                score_id=f"{trace.id}-localization.{metric_name}" if getattr(trace, "id", None) else None,
            )
        # surface telemetry in metadata for downstream consumers
        if trace:
            try:
                trace.metadata = {**getattr(trace, "metadata", {}), "telemetry": telemetry.model_dump(exclude_none=True)}
            except Exception:
                pass
        return prediction, eval_record.metrics, task.evidence, materialization_result
