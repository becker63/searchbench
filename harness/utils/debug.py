from __future__ import annotations

from typing import Any, Mapping


def _truncate(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_iteration_debug(iter_idx: int, result: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    """
    Pretty-print iteration details for humans.
    """
    print(f"\n=== ITERATION {iter_idx} ===")

    ic = result.get("iterative_context") or {}
    jc = result.get("jcodemunch") or {}

    # Tool usage from raw observations
    raw_ic = ic.get("raw") if isinstance(ic, Mapping) else {}
    observations = raw_ic.get("observations") if isinstance(raw_ic, Mapping) else []
    if observations:
        print("Tools called (iterative-context):")
        for obs in observations:
            if not isinstance(obs, Mapping):
                continue
            tool = obs.get("tool", "unknown")
            payload = obs.get("result") or obs.get("error") or ""
            summary = _truncate(str(payload))
            print(f"  - {tool}: {summary}")
    else:
        print("Tools called (iterative-context): none")

    # Exploration stats
    ic_nodes = ic.get("node_count") if isinstance(ic, Mapping) else None
    jc_nodes = jc.get("node_count") if isinstance(jc, Mapping) else None
    print(f"Exploration stats: ic_nodes={ic_nodes}, jc_nodes={jc_nodes}")

    # Metrics of interest
    score = metrics.get("score")
    coverage_delta = metrics.get("coverage_delta")
    efficiency_ic = metrics.get("efficiency_ic")
    redundancy_ic = metrics.get("redundancy_ic")
    print(
        f"Metrics: score={score}, coverage_delta={coverage_delta}, "
        f"efficiency_ic={efficiency_ic}, redundancy_ic={redundancy_ic}"
    )


def print_final_summary(history: list[Mapping[str, Any]]) -> None:
    """
    Summarize run history in a compact, human-friendly way.
    """
    if not history:
        print("No iterations completed.")
        return

    scores: list[float] = []
    coverage: list[int] = []
    efficiency: list[float] = []

    for entry in history:
        if not isinstance(entry, Mapping):
            continue
        metrics = entry.get("metrics", entry) if isinstance(entry, Mapping) else entry
        if not isinstance(metrics, Mapping):
            continue
        score_val = metrics.get("score")
        coverage_val = metrics.get("coverage_delta")
        eff_val = metrics.get("efficiency_ic")
        try:
            scores.append(float(score_val))
        except (TypeError, ValueError):
            pass
        try:
            coverage.append(int(coverage_val))
        except (TypeError, ValueError):
            pass
        try:
            efficiency.append(float(eff_val))
        except (TypeError, ValueError):
            pass

    total = len(history)
    last_score = scores[-1] if scores else None
    best_score = max(scores) if scores else None
    avg_score = sum(scores) / len(scores) if scores else None
    max_coverage = max(coverage) if coverage else None
    max_eff = max(efficiency) if efficiency else None

    print(f"Iterations completed: {total}")
    print(f"Last score: {last_score}")
    print(f"Best score: {best_score}")
    print(f"Average score: {avg_score}")
    print(f"Max coverage_delta: {max_coverage}")
    print(f"Max efficiency_ic: {max_eff}")

    if len(scores) >= 2:
        trend = "improving" if scores[-1] > scores[0] else "regressing" if scores[-1] < scores[0] else "flat"
        print(f"Trend: {trend}")
        print(f"Score history: {', '.join(f'{s:.3f}' for s in scores)}")
    elif scores:
        print(f"Score history: {scores[0]:.3f}")
    else:
        print("Score history: unavailable")
