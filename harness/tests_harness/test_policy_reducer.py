from __future__ import annotations

from harness.localization.models import LCATaskIdentity
from harness.localization.scoring_models import TaskScoreSummary
from harness.localization.token_usage import TokenUsage, TokenUsageRecord
from harness.telemetry.policy_reducer import (
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


def _score(value: float, **components: float) -> TaskScoreSummary:
    return TaskScoreSummary(
        task_id="score-task",
        composed_score=value,
        available=True,
        component_scores=dict(components) or {"gold_hop": value},
        visible_component_scores=dict(components) or {"gold_hop": value},
        component_availability={name: True for name in (components or {"gold_hop": value})},
    )


def test_equal_repo_weighting_and_order_independence():
    score_summary = _score(1.0)
    tasks = [
        build_task_input(
            identity=_identity("o1/r1"),
            task_id="t1",
            score_summary=score_summary,
            baseline_usage=_usage(100),
            candidate_usage=_usage(50),
        ),
        build_task_input(
            identity=_identity("o1/r1"),
            task_id="t2",
            score_summary=score_summary,
            baseline_usage=_usage(100),
            candidate_usage=_usage(40),
        ),
        build_task_input(
            identity=_identity("o2/r2"),
            task_id="t3",
            score_summary=score_summary,
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
    task = build_task_input(
        identity=_identity("o3/r3"),
        task_id="t_missing",
        score_summary=_score(1.0),
        baseline_usage=_usage(100),
        candidate_usage=TokenUsageRecord(available=False, usage=None),
    )
    repo_result = reduce_repo([task])
    assert repo_result.score is None
    assert repo_result.missing_candidate_tokens == 1
    assert repo_result.valid_count == 0


def test_quality_floor_drops_zero_score_tasks():
    task = build_task_input(
        identity=_identity("o4/r4"),
        task_id="t_zero",
        score_summary=_score(0.0),
        baseline_usage=_usage(50),
        candidate_usage=_usage(25),
    )
    repo_result = reduce_repo([task])
    assert repo_result.valid_count == 0
    assert repo_result.dropped_for_quality == 1


def test_token_savings_not_rewarded_when_quality_fails():
    task = build_task_input(
        identity=_identity("o5/r5"),
        task_id="t_low_quality",
        score_summary=_score(0.1),
        baseline_usage=_usage(100),
        candidate_usage=_usage(1),
    )
    repo_result = reduce_repo([task])
    assert repo_result.valid_count == 0
    assert repo_result.score is None
    assert repo_result.dropped_for_quality == 1


def test_named_component_floor_guard():
    guard = PolicyQualityGuard(score_floors={"gold_hop": 1.0})
    task = build_task_input(
        identity=_identity("o6/r6"),
        task_id="t_component_guard",
        score_summary=_score(0.9, gold_hop=0.9),
        baseline_usage=_usage(100),
        candidate_usage=_usage(10),
        quality_guard=guard,
    )
    repo_result = reduce_repo([task])
    assert repo_result.valid_count == 0
    assert repo_result.dropped_for_quality == 1
