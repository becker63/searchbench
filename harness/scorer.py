from __future__ import annotations

"""
Leaf operation: deterministic scoring/metrics.
No entrypoint or session policy logic.
"""

from collections.abc import Mapping
from typing import Dict, cast


def _obs_count(bucket: Mapping[str, object] | None) -> int:
    if not isinstance(bucket, Mapping):
        return 0
    obs = bucket.get("observations")
    return len(obs) if isinstance(obs, list) else 0


def score(result: Mapping[str, object]) -> Dict[str, object]:
    ic_bucket = cast(Mapping[str, object] | None, result.get("iterative_context"))
    jc_bucket = cast(Mapping[str, object] | None, result.get("jcodemunch"))
    ic_obs = _obs_count(ic_bucket)
    jc_obs = _obs_count(jc_bucket)
    coverage_delta = ic_obs - jc_obs
    score_val = float(coverage_delta)
    return {
        "ic_nodes": ic_obs,
        "jc_nodes": jc_obs,
        "coverage_delta": coverage_delta,
        "score": score_val,
    }
