from __future__ import annotations

from typing import List, Optional

from .models import (
    LCATaskIdentity,
    LocalizationDatasetInfo,
    LocalizationEvidence,
    LocalizationMaterialization,
    LocalizationRepoInfo,
    LocalizationTelemetryEnvelope,
)
from .scoring_models import TaskScoreSummary


def build_localization_telemetry(
    identity: LCATaskIdentity,
    score_summary: TaskScoreSummary,
    changed_files_count: Optional[int] = None,
    dataset_source: Optional[str] = None,
    repo_language: Optional[str] = None,
    repo_license: Optional[str] = None,
    evidence: Optional[LocalizationEvidence] = None,
    materialization_events: Optional[List[str]] = None,
) -> LocalizationTelemetryEnvelope:
    """
    Telemetry payload for file-path localization runs.
    Includes required LCA metadata and excludes legacy snippet/ranking fields.
    """
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
        language=repo_language,
        license=repo_license,
    )
    evidence_block = evidence
    materialization_block = LocalizationMaterialization(events=list(materialization_events)) if materialization_events else None
    return LocalizationTelemetryEnvelope(
        identity=identity.task_id(),
        dataset=dataset_block,
        dataset_source=dataset_source,
        repo=repo_block,
        score_summary=score_summary,
        changed_files_count=changed_files_count,
        evidence=evidence_block,
        materialization_events=materialization_block,
        repo_language=repo_language,
        repo_license=repo_license,
    )
