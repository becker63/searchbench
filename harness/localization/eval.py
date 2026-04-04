from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, cast

from .models import LCAGold, LCAPrediction, LCATaskIdentity
from .scoring import score_file_localization


def localization_eval_identity(identity: LCATaskIdentity) -> str:
    """
    Build a baseline/eval identity string keyed on dataset/config/split + repo_owner/name + base_sha + issue/pull key.
    """
    return identity.task_id()


def build_file_localization_eval_record(
    identity: LCATaskIdentity,
    prediction: LCAPrediction | Mapping[str, object],
    gold: LCAGold | Mapping[str, object],
    evidence: Optional[Mapping[str, List[str]]] = None,
    repo_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Build a canonical evaluation record for file-path-only predictions against LCA gold.

    Evidence (if provided) is kept separate from the primary metrics/prediction and should
    be used only for diagnostics.
    """
    metrics = score_file_localization(prediction, gold)
    if isinstance(prediction, Mapping):
        predicted_files_raw = prediction.get("predicted_files", [])
        predicted_files = [str(p) for p in cast(Iterable[object], predicted_files_raw)]
    else:
        predicted_files = prediction.normalized_predicted_files()

    if isinstance(gold, Mapping):
        changed_files_raw = gold.get("changed_files", [])
        changed_files = [str(p) for p in cast(Iterable[object], changed_files_raw)]
    else:
        changed_files = gold.normalized_changed_files()

    record: MutableMapping[str, object] = {
        "identity": localization_eval_identity(identity),
        "dataset": {
            "name": identity.dataset_name,
            "config": identity.dataset_config,
            "split": identity.dataset_split,
        },
        "repo": {
            "owner": identity.repo_owner,
            "name": identity.repo_name,
            "base_sha": identity.base_sha,
            "issue_url": identity.issue_url,
            "pull_url": identity.pull_url,
        },
        "prediction": {"predicted_files": predicted_files},
        "gold": {"changed_files": changed_files},
        "metrics": metrics,
    }
    if evidence:
        record["evidence"] = evidence
    if repo_path:
        record["repo_path"] = str(repo_path)
    return dict(record)
