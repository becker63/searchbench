from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    evidence: Optional["LocalizationEvidence"] = Field(default=None, description="Optional per-file evidence kept separate from predictions")

    def canonical_prediction_shape(self) -> str:
        """
        Describe the canonical prediction contract: file paths only.
        """
        return "predicted_files: list[str]"


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
