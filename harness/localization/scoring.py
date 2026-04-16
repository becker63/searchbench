"""Localization adapter that derives score context and delegates to scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, cast

from harness.scoring import ScoreBundle, ScoreConfig, ScoreContext, ScoreEngine
from harness.scoring.token_usage import TokenUsageRecord

from .models import LCAGold, LCAPrediction, canonicalize_paths
from .static_graph import (
    GraphStore,
    compute_hop_distance_summary,
    extract_anchors,
    ingest_repo,
    raw_tree_to_graph,
    resolve_anchors,
)
from .static_graph.store import debug_summary


def _normalize_prediction(prediction: LCAPrediction | Mapping[str, object]) -> tuple[str, ...]:
    if isinstance(prediction, Mapping):
        predicted_files_raw = prediction.get("predicted_files", [])
        return tuple(canonicalize_paths([str(p) for p in cast(Iterable[object], predicted_files_raw)]))
    return tuple(prediction.normalized_predicted_files())


def _normalize_gold(gold: LCAGold | Mapping[str, object]) -> tuple[str, ...]:
    if isinstance(gold, Mapping):
        changed_files_raw = gold.get("changed_files", [])
        return tuple(canonicalize_paths([str(p) for p in cast(Iterable[object], changed_files_raw)]))
    return tuple(gold.normalized_changed_files())


def _build_store(repo_path: Path) -> tuple[GraphStore | None, dict[str, object]]:
    try:
        tree = ingest_repo(repo_path)
        graph = raw_tree_to_graph(tree)
        store = GraphStore(graph)
        return store, {
            "static_graph_available": True,
            "static_graph_file_count": len(tree.files),
            "static_graph_node_count": debug_summary(store)["nodes"],
            "static_graph_edge_count": debug_summary(store)["edges"],
        }
    except Exception as exc:  # noqa: BLE001
        return None, {
            "static_graph_available": False,
            "static_graph_error": type(exc).__name__,
            "static_graph_error_message": str(exc),
        }


def _resolve_file_nodes(store: GraphStore, files: Iterable[str]) -> dict[str, tuple[str, ...]]:
    resolved: dict[str, tuple[str, ...]] = {}
    for path in files:
        nodes = store.find_file(path) or store.find_symbol(path)
        resolved[path] = tuple(nodes)
    return resolved


def _derive_token_count(token_usage: TokenUsageRecord | None) -> float | None:
    if token_usage is None:
        return None
    return token_usage.normalized_total()


def build_localization_score_context(
    *,
    prediction: LCAPrediction | Mapping[str, object],
    gold: LCAGold | Mapping[str, object],
    repo_path: Path,
    anchor_text: str,
    token_usage: TokenUsageRecord | None,
) -> ScoreContext:
    predicted_files = _normalize_prediction(prediction)
    gold_files = _normalize_gold(gold)
    previous_total_token_count = _derive_token_count(token_usage)
    store, graph_diagnostics = _build_store(repo_path)
    base_diagnostics: dict[str, object] = {
        **graph_diagnostics,
        "predicted_files_count": len(predicted_files),
        "gold_files_count": len(gold_files),
        "token_usage_available": bool(token_usage and token_usage.available),
        "previous_total_token_count_available": previous_total_token_count is not None,
    }
    if store is None:
        return ScoreContext(
            predicted_files=predicted_files,
            gold_files=gold_files,
            previous_total_token_count=previous_total_token_count,
            repo_path=str(repo_path),
            diagnostics=base_diagnostics,
        )

    resolved_prediction_nodes = _resolve_file_nodes(store, predicted_files)
    resolved_gold_nodes = _resolve_file_nodes(store, gold_files)
    prediction_nodes = {
        path: list(nodes) for path, nodes in resolved_prediction_nodes.items()
    }
    gold_anchor_nodes = [
        node for nodes in resolved_gold_nodes.values() for node in nodes[:1]
    ]
    gold_hops = compute_hop_distance_summary(store, gold_anchor_nodes, prediction_nodes)

    anchors = extract_anchors(anchor_text or "")
    issue_anchor_nodes = tuple(resolve_anchors(store, anchors))
    issue_hops = compute_hop_distance_summary(store, issue_anchor_nodes, prediction_nodes)
    unresolved_predictions = tuple(
        path for path, nodes in resolved_prediction_nodes.items() if not nodes
    )
    unresolved_gold_files = tuple(
        path for path, nodes in resolved_gold_nodes.items() if not nodes
    )

    return ScoreContext(
        predicted_files=predicted_files,
        gold_files=gold_files,
        gold_min_hops=gold_hops.min_hop,
        issue_min_hops=issue_hops.min_hop,
        per_prediction_hops=gold_hops.hop_by_prediction,
        previous_total_token_count=previous_total_token_count,
        resolved_prediction_nodes=resolved_prediction_nodes,
        resolved_gold_nodes=resolved_gold_nodes,
        issue_anchor_nodes=issue_anchor_nodes,
        unresolved_predictions=unresolved_predictions,
        unresolved_gold_files=unresolved_gold_files,
        repo_path=str(repo_path),
        diagnostics={
            **base_diagnostics,
            "issue_anchor_count": len(issue_anchor_nodes),
            "gold_anchor_count": len(gold_anchor_nodes),
            "resolved_prediction_count": sum(
                1 for nodes in resolved_prediction_nodes.values() if nodes
            ),
            "resolved_gold_count": sum(
                1 for nodes in resolved_gold_nodes.values() if nodes
            ),
            "unresolved_predictions": list(unresolved_predictions),
            "unresolved_gold_files": list(unresolved_gold_files),
            "gold_min_hops_available": gold_hops.min_hop is not None,
            "issue_min_hops_available": issue_hops.min_hop is not None,
        },
    )


def score_localization(
    *,
    prediction: LCAPrediction | Mapping[str, object],
    gold: LCAGold | Mapping[str, object],
    repo_path: Path,
    anchor_text: str,
    token_usage: TokenUsageRecord | None,
    config: ScoreConfig | None = None,
) -> ScoreBundle:
    context = build_localization_score_context(
        prediction=prediction,
        gold=gold,
        repo_path=repo_path,
        anchor_text=anchor_text,
        token_usage=token_usage,
    )
    return ScoreEngine(config=config).evaluate(context)


__all__ = ["build_localization_score_context", "score_localization"]
