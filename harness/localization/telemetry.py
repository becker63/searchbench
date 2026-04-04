from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from .models import LCATaskIdentity


def build_localization_telemetry(
    identity: LCATaskIdentity,
    metrics: Mapping[str, float],
    changed_files_count: Optional[int] = None,
    repo_language: Optional[str] = None,
    repo_license: Optional[str] = None,
    evidence: Optional[Mapping[str, List[str]]] = None,
    materialization_events: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Telemetry payload for file-path localization runs.
    Includes required LCA metadata and excludes legacy snippet/ranking fields.
    """
    payload: Dict[str, object] = {
        "identity": identity.task_id(),
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
            "language": repo_language,
            "license": repo_license,
        },
        "changed_files_count": changed_files_count,
        "metrics": dict(metrics),
    }
    if evidence:
        payload["evidence"] = evidence
    if materialization_events:
        payload["materialization_events"] = materialization_events
    return payload
