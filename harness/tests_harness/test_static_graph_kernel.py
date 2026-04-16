from __future__ import annotations

import math
from pathlib import Path

import pytest

from harness.localization.models import LCAGold, LCAPrediction
from harness.localization.scoring import build_localization_score_context, score_localization
from harness.localization.static_graph import (
    GraphStore,
    compute_hop_distance_summary,
    extract_anchors,
    raw_tree_to_graph,
    resolve_anchors,
)
from harness.localization.static_graph.raw_tree import RawEdge, RawFile, RawFunction, RawTree
from harness.scoring.token_usage import TokenUsage, TokenUsageRecord


def test_raw_tree_normalization_and_hops():
    tree = RawTree(
        files=[
            RawFile(
                path="src/a.py",
                functions=[RawFunction(id="src/a.py:foo", name="foo", file="src/a.py")],
                edges=[
                    RawEdge(source="src/a.py", target="src/b.py", kind="contains"),
                    RawEdge(source="src/a.py:foo", target="bar", kind="call"),
                ],
            ),
            RawFile(path="src/b.py"),
        ]
    )
    store = GraphStore(raw_tree_to_graph(tree))
    anchors = resolve_anchors(store, ["src/a.py"])
    prediction_nodes = {"src/b.py": store.find_file("src/b.py")}
    hop_result = compute_hop_distance_summary(store, anchors, prediction_nodes)
    assert hop_result.hop_by_prediction["src/b.py"] == 1
    assert hop_result.min_hop == 1


def test_anchor_extraction_order_and_dedup():
    text = "Fix in `pkg/mod.py` near src/util/helper.py and class FooBar in code ```\n def baz_snake():\n pass\n```"
    anchors = extract_anchors(text)
    assert anchors[0] == "pkg/mod.py"
    # Slashy tokens are captured; partials may appear due to regex boundaries.
    assert any("util" in a for a in anchors)
    assert "FooBar" in anchors
    assert "baz_snake" in anchors
    # ensure no duplicates
    assert len(anchors) == len(set(anchors))


def test_score_localization_uses_static_graph_context(monkeypatch, tmp_path: Path):
    # Stub ingest_repo to avoid llm-tldr dependency in test
    from harness import localization as loc_mod
    import harness.localization.scoring as scoring_mod

    tree = RawTree(
        files=[
            RawFile(
                path="repo/file_a.py",
                edges=[RawEdge(source="repo/file_a.py", target="repo/file_b.py", kind="contains")],
            ),
            RawFile(path="repo/file_b.py"),
        ]
    )

    def fake_ingest(_repo: Path) -> RawTree:
        return tree

    monkeypatch.setattr(scoring_mod, "ingest_repo", fake_ingest)
    monkeypatch.setattr(loc_mod.static_graph, "ingest_repo", fake_ingest)

    prediction = LCAPrediction(predicted_files=["repo/file_b.py"])
    gold = LCAGold(changed_files=["repo/file_a.py"])
    usage = TokenUsageRecord(available=True, usage=TokenUsage(total_tokens=100))
    ctx = build_localization_score_context(
        prediction=prediction,
        gold=gold,
        repo_path=tmp_path,
        anchor_text="`repo/file_a.py`",
        token_usage=usage,
    )
    assert ctx.gold_min_hops == 1
    assert ctx.issue_min_hops == 1
    assert ctx.previous_total_token_count == 100
    assert ctx.per_prediction_hops["repo/file_b.py"] == 1

    bundle = score_localization(
        prediction=prediction,
        gold=gold,
        repo_path=tmp_path,
        anchor_text="`repo/file_a.py`",
        token_usage=usage,
    )
    assert bundle.results["gold_hop"].value == pytest.approx(0.5)
    assert bundle.results["issue_hop"].value == pytest.approx(0.5)
    assert bundle.results["token_efficiency"].value == pytest.approx(1.0 / (1.0 + math.log1p(100)))
    assert bundle.composed_score == pytest.approx(0.5)


def test_score_context_resolves_real_ingest_repo_relative_paths(monkeypatch, tmp_path: Path):
    import harness.localization.static_graph.ingest as ingest_mod

    repo = tmp_path / "repo"
    source = repo / "src" / "app.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run():\n    return 1\n")

    monkeypatch.setattr(ingest_mod, "extract_file", lambda path: {"functions": []})
    monkeypatch.setattr(ingest_mod, "get_imports", lambda path: [])
    monkeypatch.setattr(ingest_mod, "ModuleInfo", None)
    monkeypatch.setattr(ingest_mod, "FunctionInfo", None)
    monkeypatch.setattr(ingest_mod, "ImportInfo", None)

    ctx = build_localization_score_context(
        prediction=LCAPrediction(predicted_files=["src/app.py"]),
        gold=LCAGold(changed_files=["src/app.py"]),
        repo_path=repo,
        anchor_text="Failure points at `src/app.py`.",
        token_usage=TokenUsageRecord(
            available=True,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        ),
    )

    assert ctx.resolved_prediction_nodes["src/app.py"] == ("src/app.py",)
    assert ctx.resolved_gold_nodes["src/app.py"] == ("src/app.py",)
    assert ctx.gold_min_hops == 0
    assert ctx.issue_min_hops == 0
    assert ctx.previous_total_token_count == 15
    assert ctx.diagnostics["static_graph_available"] is True
    assert ctx.diagnostics["resolved_prediction_count"] == 1
    assert ctx.diagnostics["resolved_gold_count"] == 1

    bundle = score_localization(
        prediction=LCAPrediction(predicted_files=["src/app.py"]),
        gold=LCAGold(changed_files=["src/app.py"]),
        repo_path=repo,
        anchor_text="Failure points at `src/app.py`.",
        token_usage=TokenUsageRecord(
            available=True,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        ),
    )
    assert bundle.results["gold_hop"].available is True
    assert bundle.results["issue_hop"].available is True
    assert bundle.results["token_efficiency"].available is True
    assert bundle.composed_score == pytest.approx(1.0)
    assert bundle.diagnostics["resolved_prediction_count"] == 1
    assert bundle.diagnostics["resolved_gold_count"] == 1


def test_score_localization_missing_token_usage_keeps_default_composed_score(monkeypatch, tmp_path: Path):
    import harness.localization.scoring as scoring_mod

    tree = RawTree(
        files=[
            RawFile(
                path="repo/file_a.py",
                edges=[RawEdge(source="repo/file_a.py", target="repo/file_b.py", kind="contains")],
            ),
            RawFile(path="repo/file_b.py"),
        ]
    )

    def fake_ingest(_repo: Path) -> RawTree:
        return tree

    monkeypatch.setattr(scoring_mod, "ingest_repo", fake_ingest)

    bundle = score_localization(
        prediction=LCAPrediction(predicted_files=["repo/file_b.py"]),
        gold=LCAGold(changed_files=["repo/file_a.py"]),
        repo_path=tmp_path,
        anchor_text="`repo/file_a.py`",
        token_usage=TokenUsageRecord(available=False),
    )

    assert bundle.results["token_efficiency"].available is False
    assert bundle.results["token_efficiency"].value is None
    assert bundle.composed_score == pytest.approx(0.5)
