from __future__ import annotations

from typing import Any, Dict


def _obs_nodes(result: Dict[str, Any]) -> list[Any]:
    observations = result.get("observations")
    if isinstance(observations, list):
        return observations
    return []


def extract_nodes_from_observation(obs: Dict[str, Any]) -> list[Any]:
    result = obs.get("result")
    if not isinstance(result, dict):
        return []

    if "nodes" in result and isinstance(result["nodes"], list):
        return result["nodes"]

    if "symbols" in result and isinstance(result["symbols"], list):
        return result["symbols"]

    if "id" in result:
        return [result]

    return []


def project_observations(observations: list[Dict[str, Any]]) -> Dict[str, Any]:
    nodes: list[Any] = []
    seen_ids: set[str] = set()

    for obs in observations:
        extracted = extract_nodes_from_observation(obs)
        for node in extracted:
            node_id = None
            if isinstance(node, dict):
                node_id = str(node.get("id") or node.get("symbol_id") or "")
            if node_id:
                if node_id in seen_ids:
                    continue
                seen_ids.add(node_id)
            nodes.append(node)

    return {
        "nodes": nodes,
        "edges": [],
        "node_count": len(nodes),
        "edge_count": 0,
    }


def normalize_iterative_context(result: Dict[str, Any]) -> Dict[str, Any]:
    if "observations" in result:
        obs = _obs_nodes(result)
        projected = project_observations(obs)
        return {**projected, "raw": result}

    graph = result.get("graph", result)
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    node_count = len(nodes) if isinstance(nodes, list) else 0
    edge_count = len(edges) if isinstance(edges, list) else 0
    return {
        "nodes": nodes if isinstance(nodes, list) else [],
        "edges": edges if isinstance(edges, list) else [],
        "node_count": node_count,
        "edge_count": edge_count,
        "raw": result,
    }


def normalize_jcodemunch(result: Dict[str, Any]) -> Dict[str, Any]:
    if "observations" in result:
        obs = _obs_nodes(result)
        projected = project_observations(obs)
        return {**projected, "raw": result}

    bundle = result.get("bundle", result)
    symbols = bundle.get("symbols", []) if isinstance(bundle, dict) else []
    node_count = len(symbols) if isinstance(symbols, list) else 0
    return {
        "nodes": symbols if isinstance(symbols, list) else [],
        "edges": [],
        "node_count": node_count,
        "edge_count": 0,
        "raw": result,
    }
