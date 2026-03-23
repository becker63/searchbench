from __future__ import annotations

from collections.abc import Mapping, MutableMapping


def _ic_node_count(result: Mapping[str, object]) -> int:
    graph_obj = result.get("iterative_context")
    graph = graph_obj.get("graph") if isinstance(graph_obj, MutableMapping) else {}
    nodes = graph.get("nodes") if isinstance(graph, MutableMapping) else None
    return len(nodes) if isinstance(nodes, list) else 0


def _jc_node_count(result: Mapping[str, object]) -> int:
    jc_obj = result.get("jcodemunch")
    bundle = jc_obj.get("bundle") if isinstance(jc_obj, MutableMapping) else {}
    if not isinstance(bundle, MutableMapping):
        return 0
    if "symbols" in bundle and isinstance(bundle["symbols"], list):
        return len(bundle["symbols"])
    # Single-symbol responses still count as one if no error is present.
    if "error" in bundle:
        return 0
    return 1 if bundle else 0


def score(result: Mapping[str, object]) -> dict[str, float | int]:
    """
    Compute deterministic metrics comparing iterative-context and jcodemunch outputs.

    Coverage is approximated by node counts; the score rewards greater coverage
    while gently penalizing overly large graphs.
    """
    ic_nodes = _ic_node_count(result)
    jc_nodes = _jc_node_count(result)

    coverage_delta = ic_nodes - jc_nodes
    efficiency_penalty = 0.05 * ic_nodes
    total_score = float(coverage_delta - efficiency_penalty)

    return {
        "ic_nodes": int(ic_nodes),
        "jc_nodes": int(jc_nodes),
        "coverage_delta": int(coverage_delta),
        "score": total_score,
    }
