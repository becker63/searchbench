from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _normalize_path_list(paths: List[str]) -> List[str]:
    """
    Canonicalize a list of file paths into a stable, duplicate-free, trimmed list.
    """
    return sorted({p.strip() for p in paths if p and p.strip()})


class LCATaskIdentity(BaseModel):
    """
    Identity for an LCA bug-localization task.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    dataset_config: str
    dataset_split: str
    repo_owner: str
    repo_name: str
    base_sha: str
    issue_url: Optional[str] = None
    pull_url: Optional[str] = None

    def task_id(self) -> str:
        """
        Deterministic ID built from dataset/config/split, repo owner/name, base_sha, and issue/pull URL.
        """
        repo = f"{self.repo_owner.lower()}/{self.repo_name.lower()}@{self.base_sha}"
        issue_key = self.issue_url or self.pull_url or "unknown-issue"
        return ":".join(
            [
                self.dataset_name,
                self.dataset_config,
                self.dataset_split,
                repo,
                issue_key,
            ]
        )


class LCAContext(BaseModel):
    """
    Runtime context delivered to the localization agent.
    """

    model_config = ConfigDict(extra="forbid")

    issue_title: str
    issue_body: str
    diff_url: Optional[str] = None
    diff: Optional[str] = None
    head_sha: Optional[str] = None
    repo_language: Optional[str] = None
    repo_languages: List[str] = Field(default_factory=list)
    repo_license: Optional[str] = None
    repo_stars: Optional[int] = None

    def agent_payload(self) -> Dict[str, object]:
        """
        Agent-facing, localization-native payload: issue text plus repo/diff context.
        """
        return {
            "issue_title": self.issue_title,
            "issue_body": self.issue_body,
            "diff_url": self.diff_url,
            "diff": self.diff,
            "head_sha": self.head_sha,
            "repo_language": self.repo_language,
            "repo_languages": self.repo_languages,
            "repo_license": self.repo_license,
            "repo_stars": self.repo_stars,
        }


class LCAGold(BaseModel):
    """
    Canonical gold for LCA bug localization: changed_files.
    """

    model_config = ConfigDict(extra="forbid")

    changed_files: List[str]
    changed_files_count: Optional[int] = None
    changed_files_exts: Optional[Dict[str, int]] = None

    def normalized_changed_files(self) -> List[str]:
        return _normalize_path_list(self.changed_files)


class LCAPrediction(BaseModel):
    """
    Canonical prediction: file paths only (no rank/confidence/score).
    """

    model_config = ConfigDict(extra="forbid")

    predicted_files: List[str]

    def normalized_predicted_files(self) -> List[str]:
        return _normalize_path_list(self.predicted_files)


class LCATask(BaseModel):
    """
    LCA bug-localization task model grouping identity, runtime context, gold, and optional diagnostics.
    """

    model_config = ConfigDict(extra="forbid")

    identity: LCATaskIdentity
    context: LCAContext
    gold: LCAGold
    evidence: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Optional per-file evidence kept separate from predictions"
    )

    def canonical_prediction_shape(self) -> str:
        """
        Describe the canonical prediction contract: file paths only.
        """
        return "predicted_files: list[str]"

