from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Optional, cast

from harness.localization.models import (
    LCAGold,
    LCAPrediction,
    LCATaskIdentity,
    LocalizationDatasetInfo,
    LocalizationEvalRecord,
    LocalizationEvidence,
    LocalizationGold,
    LocalizationMetrics,
    LocalizationPrediction,
    LocalizationRepoInfo,
)
from harness.localization.scoring import score_file_localization


def localization_eval_identity(identity: LCATaskIdentity) -> str:
    """
    Build a baseline/eval identity string keyed on dataset/config/split + repo_owner/name + base_sha + issue/pull key.
    """
    return identity.task_id()


def _coerce_paths(value: Iterable[object]) -> List[str]:
    return [str(p) for p in value if isinstance(p, (str, Path)) or hasattr(p, "__fspath__")]


def _normalize_prediction(prediction: LCAPrediction | Mapping[str, object]) -> List[str]:
    if isinstance(prediction, Mapping):
        predicted_files_raw = prediction.get("predicted_files", [])
        return _coerce_paths(cast(Iterable[object], predicted_files_raw))
    return prediction.normalized_predicted_files()


def _normalize_gold(gold: LCAGold | Mapping[str, object]) -> List[str]:
    if isinstance(gold, Mapping):
        changed_files_raw = gold.get("changed_files", [])
        return _coerce_paths(cast(Iterable[object], changed_files_raw))
    return gold.normalized_changed_files()


def build_file_localization_eval_record(
    identity: LCATaskIdentity,
    prediction: LCAPrediction | Mapping[str, object],
    gold: LCAGold | Mapping[str, object],
    evidence: Optional[LocalizationEvidence] = None,
    repo_path: Optional[Path] = None,
) -> LocalizationEvalRecord:
    """
    Build a canonical evaluation record for file-path-only predictions against LCA gold.

    Evidence (if provided) is kept separate from the primary metrics/prediction and should
    be used only for diagnostics.
    """
    metrics = score_file_localization(prediction, gold)
    predicted_files = _normalize_prediction(prediction)
    changed_files = _normalize_gold(gold)

    dataset_block = LocalizationDatasetInfo(
        name=identity.dataset_name,
        config=identity.dataset_config,
        split=identity.dataset_split,
    )
    repo_block = LocalizationRepoInfo(
        owner=identity.repo_owner,
        name=identity.repo_name,
        base_sha=identity.base_sha,
        issue_url=identity.issue_url,
        pull_url=identity.pull_url,
    )
    evidence_block = evidence
    repo_path_str = str(repo_path) if repo_path else None
    return LocalizationEvalRecord(
        identity=localization_eval_identity(identity),
        dataset=dataset_block,
        repo=repo_block,
        prediction=LocalizationPrediction(predicted_files=predicted_files),
        gold=LocalizationGold(changed_files=changed_files),
        metrics=metrics if isinstance(metrics, LocalizationMetrics) else LocalizationMetrics.from_mapping(metrics),
        evidence=evidence_block,
        repo_path=repo_path_str,
    )
