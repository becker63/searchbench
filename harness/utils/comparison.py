from __future__ import annotations

from typing import Mapping

DEFAULT_WEIGHTS: dict[str, float] = {
    "score": 1.0,
    "ic_nodes": 0.2,
    "jc_nodes": 0.2,
}


def _as_float(val: object) -> float:
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return float(val)
    return 0.0


def build_comparison(
    ic_metrics: Mapping[str, object],
    jc_metrics: Mapping[str, object],
    weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    deltas: dict[str, float] = {}
    weighted = 0.0
    for key, weight in w.items():
        delta = _as_float(ic_metrics.get(key, 0.0)) - _as_float(jc_metrics.get(key, 0.0))
        deltas[key] = delta
        weighted += weight * delta
    return {
        "ic_metrics": dict(ic_metrics),
        "jc_metrics": dict(jc_metrics),
        "deltas": deltas,
        "weighted_score": weighted,
    }


def format_comparison_summary(comp: Mapping[str, object]) -> str:
    deltas = comp.get("deltas")
    weighted = comp.get("weighted_score")
    if not isinstance(deltas, Mapping):
        return "No baseline comparison available."
    parts = ["IC vs JC baseline:"]
    for k, v in deltas.items():
        parts.append(f"- {k}: {v:+.3f}")
    if isinstance(weighted, (int, float)):
        parts.append(f"Weighted delta: {float(weighted):+.3f}")
    return "\n".join(parts)
