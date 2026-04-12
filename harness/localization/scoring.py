from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, cast

from .models import (
    LCAGold,
    LCAPrediction,
    LocalizationMetrics,
    canonicalize_paths,
)
from .static_graph import (
    GraphStore,
    compute_hop_distance_summary,
    extract_anchors,
    ingest_repo,
    raw_tree_to_graph,
    resolve_anchors,
)
from .token_usage import TokenUsageRecord


def _as_path_set(paths: Iterable[str]) -> set[str]:
    return set(canonicalize_paths(paths))


def file_localization_metrics(predicted_files: Iterable[str], changed_files: Iterable[str]) -> LocalizationMetrics:
    """
    Compute file-set localization metrics from predicted file paths vs. gold changed_files.
    """
    pred_set = _as_path_set(predicted_files)
    gold_set = _as_path_set(changed_files)
    if not gold_set:
        return LocalizationMetrics(precision=0.0, recall=0.0, f1=0.0, hit=0.0, score=0.0)

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gold_set)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    hit = 1.0 if true_positives > 0 else 0.0
    return LocalizationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        hit=hit,
        score=f1,
    )


def score_file_localization(prediction: LCAPrediction | Mapping[str, object], gold: LCAGold | Mapping[str, object]) -> LocalizationMetrics:
    """
    Phase-1 scorer: primary metrics from file-path predictions vs. `changed_files`.
    `metrics.score` is mapped to F1 for a simple, stable scalar.
    """
    if isinstance(prediction, Mapping):
        predicted_files_raw = prediction.get("predicted_files", [])
        predicted_files = canonicalize_paths([str(p) for p in cast(Iterable[object], predicted_files_raw)])
    else:
        predicted_files = prediction.normalized_predicted_files()

    if isinstance(gold, Mapping):
        changed_files_raw = gold.get("changed_files", [])
        changed_files = canonicalize_paths([str(p) for p in cast(Iterable[object], changed_files_raw)])
    else:
        changed_files = gold.normalized_changed_files()

    return file_localization_metrics(predicted_files, changed_files)


def _distance_term(min_hop: int | None) -> float | None:
    if min_hop is None:
        return None
    return 1.0 / (1.0 + float(min_hop))


def _token_term(token_usage: TokenUsageRecord | None) -> float | None:
    if token_usage is None:
        return None
    total = token_usage.normalized_total()
    if total is None:
        return None
    return 1.0 / (1.0 + math.log1p(total))


def _combine_terms(distance_term: float | None, token_term: float | None, distance_weight: float = 0.5) -> float | None:
    terms: list[tuple[float, float]] = []
    if distance_term is not None:
        terms.append((distance_term, distance_weight))
    if token_term is not None:
        # If distance is missing, give token term full weight; otherwise share remaining weight.
        weight = 1.0 if distance_term is None else max(0.0, 1.0 - distance_weight)
        terms.append((token_term, weight))
    if not terms:
        return None
    weight_sum = sum(weight for _, weight in terms)
    if weight_sum == 0:
        return None
    return sum(val * weight for val, weight in terms) / weight_sum


def _build_store(repo_path: Path) -> GraphStore | None:
    try:
        tree = ingest_repo(repo_path)
        graph = raw_tree_to_graph(tree)
        return GraphStore(graph)
    except Exception:
        return None


def score_with_projection(
    prediction: LCAPrediction | Mapping[str, object],
    gold: LCAGold | Mapping[str, object],
    repo_path: Path,
    anchor_text: str,
    token_usage: TokenUsageRecord | None,
) -> LocalizationMetrics:
    """
    Compute legacy localization metrics plus projected hop/token scalar.
    Does not overwrite legacy metric semantics.
    """
    if isinstance(prediction, Mapping):
        predicted_files_raw = prediction.get("predicted_files", [])
        predicted_files = canonicalize_paths([str(p) for p in cast(Iterable[object], predicted_files_raw)])
    else:
        predicted_files = prediction.normalized_predicted_files()

    if isinstance(gold, Mapping):
        changed_files_raw = gold.get("changed_files", [])
        changed_files = canonicalize_paths([str(p) for p in cast(Iterable[object], changed_files_raw)])
    else:
        changed_files = gold.normalized_changed_files()

    metrics = file_localization_metrics(predicted_files, changed_files)

    store = _build_store(repo_path)
    if store is None:
        metrics.per_file_hops = {}
        return metrics

    anchors = extract_anchors(anchor_text or "")
    anchor_nodes = resolve_anchors(store, anchors)

    prediction_nodes: dict[str, list[str]] = {}
    for pred in predicted_files:
        nodes = store.find_file(pred) or store.find_symbol(pred)
        prediction_nodes[pred] = nodes

    hop_result = compute_hop_distance_summary(store, anchor_nodes, prediction_nodes)
    distance_val = _distance_term(hop_result.min_hop)
    token_val = _token_term(token_usage)
    projected = _combine_terms(distance_val, token_val)

    metrics.distance_term = distance_val
    metrics.token_term = token_val
    metrics.hop_token_score = projected
    metrics.per_file_hops = hop_result.hop_by_prediction or {}
    return metrics
