from __future__ import annotations

from harness.localization.models import LCATaskIdentity, LocalizationMetrics
from harness.localization.token_usage import TokenUsage, TokenUsageRecord
from harness.observability.policy_reducer import (
    build_task_input,
    PolicyQualityGuard,
    reduce_global,
    reduce_repo,
)


def _identity(repo: str) -> LCATaskIdentity:
    owner, name = repo.split("/")
    return LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner=owner,
        repo_name=name,
        base_sha="abc",
    )


def _usage(total: float) -> TokenUsageRecord:
    return TokenUsageRecord(available=True, usage=TokenUsage(total_tokens=total))


def test_equal_repo_weighting_and_order_independence():
    metrics = LocalizationMetrics(precision=1.0, recall=1.0, f1=1.0, hit=1.0, score=1.0)
    tasks = [
        build_task_input(
            identity=_identity("o1/r1"),
            task_id="t1",
            metrics=metrics,
            baseline_usage=_usage(100),
            candidate_usage=_usage(50),
        ),
        build_task_input(
            identity=_identity("o1/r1"),
            task_id="t2",
            metrics=metrics,
            baseline_usage=_usage(100),
            candidate_usage=_usage(40),
        ),
        build_task_input(
            identity=_identity("o2/r2"),
            task_id="t3",
            metrics=metrics,
            baseline_usage=_usage(100),
            candidate_usage=_usage(90),
        ),
    ]
    # Shuffle input order to prove determinism
    reordered = [tasks[2], tasks[0], tasks[1]]
    summary = reduce_global(reordered)
    repo_scores = {r.repo_key: r.score for r in summary.repo_results}
    # Repo1 median reduction: (0.5, 0.6) -> 0.55; Repo2: 0.1; global average (equal weight) -> (0.55 + 0.1)/2
    assert repo_scores["o1/r1@abc"] == 0.55
    assert repo_scores["o2/r2@abc"] == 0.1
    assert summary.global_score == 0.325


def test_missing_tokens_handled_explicitly():
    metrics = LocalizationMetrics(precision=1.0, recall=1.0, f1=1.0, hit=1.0, score=1.0)
    task = build_task_input(
        identity=_identity("o3/r3"),
        task_id="t_missing",
        metrics=metrics,
        baseline_usage=_usage(100),
        candidate_usage=TokenUsageRecord(available=False, usage=None),
    )
    repo_result = reduce_repo([task])
    assert repo_result.score is None
    assert repo_result.missing_candidate_tokens == 1
    assert repo_result.valid_count == 0


def test_quality_floor_drops_zero_score_tasks():
    zero_metrics = LocalizationMetrics(precision=0.0, recall=0.0, f1=0.0, hit=0.0, score=0.0)
    task = build_task_input(
        identity=_identity("o4/r4"),
        task_id="t_zero",
        metrics=zero_metrics,
        baseline_usage=_usage(50),
        candidate_usage=_usage(25),
    )
    repo_result = reduce_repo([task])
    assert repo_result.valid_count == 0
    assert repo_result.dropped_for_quality == 1


def test_token_savings_not_rewarded_when_quality_fails():
    low_metrics = LocalizationMetrics(precision=0.1, recall=0.1, f1=0.1, hit=0.0, score=0.1)
    task = build_task_input(
        identity=_identity("o5/r5"),
        task_id="t_low_quality",
        metrics=low_metrics,
        baseline_usage=_usage(100),
        candidate_usage=_usage(1),
    )
    repo_result = reduce_repo([task])
    assert repo_result.valid_count == 0
    assert repo_result.score is None
    assert repo_result.dropped_for_quality == 1


def test_optional_hit_floor_guard():
    metrics = LocalizationMetrics(precision=0.9, recall=0.9, f1=0.9, hit=0.0, score=0.9)
    guard = PolicyQualityGuard(hit_floor=1.0)
    task = build_task_input(
        identity=_identity("o6/r6"),
        task_id="t_hit_guard",
        metrics=metrics,
        baseline_usage=_usage(100),
        candidate_usage=_usage(10),
        quality_guard=guard,
    )
    repo_result = reduce_repo([task])
    assert repo_result.valid_count == 0
    assert repo_result.dropped_for_quality == 1
