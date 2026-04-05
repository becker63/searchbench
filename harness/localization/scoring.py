from __future__ import annotations

from typing import Iterable, Mapping, cast

from .models import LCAGold, LCAPrediction, LocalizationMetrics


def _as_path_set(paths: Iterable[str]) -> set[str]:
    return {p.strip() for p in paths if p and p.strip()}


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
        predicted_files = [str(p) for p in cast(Iterable[object], predicted_files_raw)]
    else:
        predicted_files = prediction.normalized_predicted_files()

    if isinstance(gold, Mapping):
        changed_files_raw = gold.get("changed_files", [])
        changed_files = [str(p) for p in cast(Iterable[object], changed_files_raw)]
    else:
        changed_files = gold.normalized_changed_files()

    return file_localization_metrics(predicted_files, changed_files)
