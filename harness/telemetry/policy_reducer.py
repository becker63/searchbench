from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from harness.localization.models import LCATaskIdentity
from harness.localization.scoring_models import TaskScoreSummary
from harness.localization.token_usage import TokenUsageRecord


def repo_key_from_identity(identity: LCATaskIdentity) -> str:
    """Deterministic repo key for reducer grouping."""
    return f"{identity.repo_owner}/{identity.repo_name}@{identity.base_sha}"


class PolicyReducerTaskInput(BaseModel):
    """Task-level reducer input."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    repo_key: str
    dataset_name: str
    dataset_config: str
    dataset_split: str
    score_summary: TaskScoreSummary
    baseline_usage: TokenUsageRecord = Field(default_factory=TokenUsageRecord)
    candidate_usage: TokenUsageRecord = Field(default_factory=TokenUsageRecord)
    quality_ok: bool = True
    quality_reasons: list[str] = Field(default_factory=list)
    token_delta: float | None = None


class PolicyQualityGuard(BaseModel):
    """Configurable quality floors for reducer eligibility."""

    model_config = ConfigDict(extra="forbid")

    score_floors: dict[str, float] = Field(default_factory=lambda: {"composed_score": 0.2})

    def evaluate(self, score_summary: TaskScoreSummary) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for score_name, floor in self.score_floors.items():
            value = (
                score_summary.composed_score
                if score_name == "composed_score"
                else score_summary.component_scores.get(score_name)
            )
            if value is None or value < floor:
                reasons.append(f"{score_name}_below_floor")
        return (len(reasons) == 0, reasons)


class RepoReducerResult(BaseModel):
    """Repo-level reducer output."""

    model_config = ConfigDict(extra="forbid")

    repo_key: str
    score: float | None
    task_count: int
    valid_count: int
    missing_baseline_tokens: int = 0
    missing_candidate_tokens: int = 0
    dropped_for_quality: int = 0


class PolicyReducerSummary(BaseModel):
    """Global reducer summary."""

    model_config = ConfigDict(extra="forbid")

    global_score: float | None
    repo_results: list[RepoReducerResult] = Field(default_factory=list)
    repo_count: int = 0
    valid_repo_count: int = 0


def _compute_token_delta(baseline: TokenUsageRecord, candidate: TokenUsageRecord) -> float | None:
    base_total = baseline.normalized_total() if baseline else None
    cand_total = candidate.normalized_total() if candidate else None
    if not baseline or not baseline.available:
        return None
    if not candidate or not candidate.available:
        return None
    if base_total is None or base_total <= 0 or cand_total is None:
        return None
    return (base_total - cand_total) / base_total


def build_task_input(
    *,
    identity: LCATaskIdentity,
    task_id: str,
    score_summary: TaskScoreSummary,
    candidate_usage: TokenUsageRecord | None,
    baseline_usage: TokenUsageRecord | None = None,
    quality_guard: PolicyQualityGuard | None = None,
) -> PolicyReducerTaskInput:
    baseline_record = baseline_usage or TokenUsageRecord(available=False, usage=None, source="baseline")
    candidate_record = candidate_usage or TokenUsageRecord(available=False, usage=None, source="candidate")
    token_delta = _compute_token_delta(baseline_record, candidate_record)
    guard = quality_guard or PolicyQualityGuard()
    quality_ok, reasons = guard.evaluate(score_summary)
    return PolicyReducerTaskInput(
        task_id=task_id,
        repo_key=repo_key_from_identity(identity),
        dataset_name=identity.dataset_name,
        dataset_config=identity.dataset_config,
        dataset_split=identity.dataset_split,
        score_summary=score_summary,
        baseline_usage=baseline_record,
        candidate_usage=candidate_record,
        quality_ok=quality_ok,
        quality_reasons=reasons,
        token_delta=token_delta,
    )


def reduce_repo(tasks: Sequence[PolicyReducerTaskInput]) -> RepoReducerResult:
    valid_deltas = [t.token_delta for t in tasks if t.quality_ok and t.token_delta is not None]
    score = median(valid_deltas) if valid_deltas else None
    return RepoReducerResult(
        repo_key=tasks[0].repo_key if tasks else "unknown",
        score=score,
        task_count=len(tasks),
        valid_count=len(valid_deltas),
        missing_baseline_tokens=sum(1 for t in tasks if not t.baseline_usage.available),
        missing_candidate_tokens=sum(1 for t in tasks if not t.candidate_usage.available),
        dropped_for_quality=sum(1 for t in tasks if not t.quality_ok),
    )


def reduce_global(tasks: Iterable[PolicyReducerTaskInput]) -> PolicyReducerSummary:
    grouped: dict[str, list[PolicyReducerTaskInput]] = defaultdict(list)
    for task in tasks:
        grouped[task.repo_key].append(task)
    repo_results = [reduce_repo(items) for items in grouped.values()]
    repo_scores = [r.score for r in repo_results if r.score is not None]
    global_score = sum(repo_scores) / len(repo_scores) if repo_scores else None
    return PolicyReducerSummary(
        global_score=global_score,
        repo_results=sorted(repo_results, key=lambda r: r.repo_key),
        repo_count=len(repo_results),
        valid_repo_count=len(repo_scores),
    )
