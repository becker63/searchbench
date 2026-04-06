from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    @property
    def id(self) -> str:
        """
        Alias for task_id to simplify referencing identity as the canonical key.
        """
        return self.task_id()


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


class LCAGold(BaseModel):
    """
    Canonical gold for LCA bug localization: changed_files.
    """

    model_config = ConfigDict(extra="forbid")

    changed_files: List[str]
    changed_files_count: Optional[int] = None
    changed_files_exts: Optional["LocalizationExtensionCounts"] = None

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
    repo: Optional[str] = None
    evidence: Optional["LocalizationEvidence"] = Field(default=None, description="Optional per-file evidence kept separate from predictions")

    def canonical_prediction_shape(self) -> str:
        """
        Describe the canonical prediction contract: file paths only.
        """
        return "predicted_files: list[str]"

    @property
    def task_id(self) -> str:
        """
        Deterministic task identifier (delegates to identity and includes repo when available).
        """
        base_id = self.identity.task_id()
        if self.repo:
            return f"{base_id}:{self.repo}"
        return base_id

    @property
    def changed_files(self) -> List[str]:
        """
        Convenience accessor for gold changed_files.
        """
        return self.gold.normalized_changed_files()


class LocalizationDatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    config: str
    split: str


class LocalizationRepoInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str
    name: str
    base_sha: str
    issue_url: Optional[str] = None
    pull_url: Optional[str] = None
    language: Optional[str] = None
    license: Optional[str] = None


class LocalizationPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_files: List[str]

    @classmethod
    def from_raw(cls, value: Sequence[str] | Mapping[str, Sequence[str]]) -> "LocalizationPrediction":
        if isinstance(value, Mapping):
            files = value.get("predicted_files", [])
            return cls(predicted_files=list(files))
        return cls(predicted_files=list(value))


class LocalizationGold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changed_files: List[str]

    @classmethod
    def from_raw(cls, value: Sequence[str] | Mapping[str, Sequence[str]]) -> "LocalizationGold":
        if isinstance(value, Mapping):
            files = value.get("changed_files", [])
            return cls(changed_files=list(files))
        return cls(changed_files=list(value))


class LocalizationEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    hints: List[str] = Field(default_factory=list)


class LocalizationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[LocalizationEvidenceItem] = Field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Sequence[str]]) -> "LocalizationEvidence":
        return cls(
            items=[
                LocalizationEvidenceItem(file=str(path), hints=[str(h) for h in hints])
                for path, hints in value.items()
            ]
        )


class LocalizationExtensionCount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extension: str
    count: int


class LocalizationExtensionCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: List[LocalizationExtensionCount] = Field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: Mapping[str, int]) -> "LocalizationExtensionCounts":
        return cls(entries=[LocalizationExtensionCount(extension=str(ext), count=int(count)) for ext, count in value.items()])


class LocalizationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: float
    recall: float
    f1: float
    hit: float
    score: float

    @classmethod
    def from_mapping(cls, metrics: Mapping[str, float]) -> "LocalizationMetrics":
        return cls(
            precision=float(metrics.get("precision", 0.0)),
            recall=float(metrics.get("recall", 0.0)),
            f1=float(metrics.get("f1", 0.0)),
            hit=float(metrics.get("hit", 0.0)),
            score=float(metrics.get("score", metrics.get("f1", 0.0))),
        )


class LocalizationMaterialization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: List[str] = Field(default_factory=list)


class LocalizationEvalRecord(BaseModel):
    """
    Canonical evaluation record for file-path localization.
    """

    model_config = ConfigDict(extra="forbid")

    identity: str
    dataset: LocalizationDatasetInfo
    repo: LocalizationRepoInfo
    prediction: LocalizationPrediction
    gold: LocalizationGold
    metrics: LocalizationMetrics
    evidence: Optional[LocalizationEvidence] = None
    repo_path: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_blocks(cls, value: object) -> object:
        if isinstance(value, Mapping):
            raw = dict(value)
            metrics_val = raw.get("metrics")
            if isinstance(metrics_val, Mapping):
                raw["metrics"] = LocalizationMetrics.from_mapping(metrics_val)
            evidence_val = raw.get("evidence")
            if isinstance(evidence_val, Mapping):
                raw["evidence"] = LocalizationEvidence.from_mapping(evidence_val)
            return raw
        return value


class LocalizationTelemetryEnvelope(BaseModel):
    """
    Structured telemetry envelope for localization runs.
    """

    model_config = ConfigDict(extra="forbid")

    identity: str
    dataset: LocalizationDatasetInfo
    dataset_source: Optional[str] = None
    repo: LocalizationRepoInfo
    metrics: LocalizationMetrics
    changed_files_count: Optional[int] = None
    evidence: Optional[LocalizationEvidence] = None
    materialization_events: Optional[LocalizationMaterialization] = None
    repo_language: Optional[str] = None
    repo_license: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_blocks(cls, value: object) -> object:
        if isinstance(value, Mapping):
            raw = dict(value)
            metrics_val = raw.get("metrics")
            if isinstance(metrics_val, Mapping):
                raw["metrics"] = LocalizationMetrics.from_mapping(metrics_val)
            evidence_val = raw.get("evidence")
            if isinstance(evidence_val, Mapping):
                raw["evidence"] = LocalizationEvidence.from_mapping(evidence_val)
            materialization_val = raw.get("materialization_events")
            if isinstance(materialization_val, list):
                raw["materialization_events"] = LocalizationMaterialization(events=list(materialization_val))
            return raw
        return value


def _as_list(value: object) -> List[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]


def normalize_lca_task(data: Mapping[str, object], defaults: tuple[str, str, str] | None = None) -> LCATask:
    """
    Coerce an arbitrary mapping (e.g., LCA dataset row) into an `LCATask`.
    `defaults` may provide (dataset_name, dataset_config, dataset_split) when absent from the row.
    """
    identity_map = data.get("identity") if isinstance(data.get("identity"), Mapping) else {}
    dataset_name = str(
        data.get("dataset_name")
        or (identity_map.get("dataset_name") if isinstance(identity_map, Mapping) else None)
        or data.get("dataset")
        or (defaults[0] if defaults else "unknown-dataset")
    )
    dataset_config = str(
        data.get("dataset_config")
        or (identity_map.get("dataset_config") if isinstance(identity_map, Mapping) else None)
        or data.get("config")
        or (defaults[1] if defaults else "unknown-config")
    )
    dataset_split = str(
        data.get("dataset_split")
        or (identity_map.get("dataset_split") if isinstance(identity_map, Mapping) else None)
        or data.get("split")
        or (defaults[2] if defaults else "unknown-split")
    )
    repo_owner = str(
        data.get("repo_owner")
        or (identity_map.get("repo_owner") if isinstance(identity_map, Mapping) else None)
        or data.get("owner")
        or data.get("repository_owner")
        or "unknown-owner"
    )
    repo_name = str(
        data.get("repo_name")
        or (identity_map.get("repo_name") if isinstance(identity_map, Mapping) else None)
        or data.get("name")
        or data.get("repository_name")
        or "unknown-repo"
    )
    base_sha = str(
        data.get("base_sha")
        or (identity_map.get("base_sha") if isinstance(identity_map, Mapping) else None)
        or data.get("sha")
        or data.get("commit")
        or "unknown-base-sha"
    )
    issue_url = (
        (identity_map.get("issue_url") if isinstance(identity_map, Mapping) else None)
        or data.get("issue_url")
        or data.get("issue")
    )
    pull_url = (
        (identity_map.get("pull_url") if isinstance(identity_map, Mapping) else None)
        or data.get("pull_url")
        or data.get("pull")
    )

    identity = LCATaskIdentity(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        repo_owner=repo_owner,
        repo_name=repo_name,
        base_sha=base_sha,
        issue_url=str(issue_url) if issue_url else None,
        pull_url=str(pull_url) if pull_url else None,
    )

    issue_title = str(data.get("issue_title") or data.get("title") or "")
    issue_body = str(data.get("issue_body") or data.get("body") or "")
    repo_stars_val = data.get("repo_stars")
    context = LCAContext(
        issue_title=issue_title,
        issue_body=issue_body,
        diff_url=str(data.get("diff_url")) if data.get("diff_url") else None,
        diff=str(data.get("diff")) if data.get("diff") else None,
        head_sha=str(data.get("head_sha")) if data.get("head_sha") else None,
        repo_language=str(data.get("repo_language")) if data.get("repo_language") else None,
        repo_languages=_as_list(data.get("repo_languages") or []),
        repo_license=str(data.get("repo_license")) if data.get("repo_license") else None,
        repo_stars=int(repo_stars_val) if isinstance(repo_stars_val, (int, float)) else None,
    )

    changed_files_raw = data.get("changed_files") or data.get("gold") or []
    gold = LCAGold(changed_files=_as_list(changed_files_raw))

    evidence_val = data.get("evidence")
    evidence = LocalizationEvidence.from_mapping(evidence_val) if isinstance(evidence_val, Mapping) else None

    repo_val = data.get("repo")
    repo_str = str(repo_val) if isinstance(repo_val, (str, Path)) else None

    return LCATask(identity=identity, context=context, gold=gold, repo=repo_str, evidence=evidence)
