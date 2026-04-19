# pyright: reportMissingTypeStubs=false
from typing import cast

from iterative_context.test_helpers import (
    build_graph,
    normalize_graph,
    run_expansion_steps,
    select_first_node,
)
from iterative_context.test_helpers.graph_dsl import GraphSpec, GraphType


def run_steps(graph: GraphType, steps: int) -> GraphType:
    final_graph, _ = run_expansion_steps(
        graph,
        max_steps=steps,
        select_node_fn=select_first_node,
    )
    return final_graph


def test_noise_injection_invariance():
    base = build_graph(
        cast(GraphSpec, {"nodes": [{"id": "A", "kind": "symbol", "state": "pending"}]})
    )
    noisy = build_graph(
        cast(
            GraphSpec,
            {
                "nodes": [
                    {"id": "A", "kind": "symbol", "state": "pending"},
                    {"id": "Z1", "kind": "symbol", "state": "pending"},
                    {"id": "Z2", "kind": "symbol", "state": "pending"},
                ]
            },
        )
    )

    base_result = run_steps(base, 3)
    noisy_result = run_steps(noisy, 3)

    norm_base = normalize_graph(base_result)
    norm_noisy = normalize_graph(noisy_result)
    filtered_noisy = {
        "nodes": [n for n in norm_noisy["nodes"] if not n["id"].startswith("Z")],
        "edges": [
            e
            for e in norm_noisy["edges"]
            if not (e["source"].startswith("Z") or e["target"].startswith("Z"))
        ],
    }

    assert norm_base == filtered_noisy


def test_order_invariance_initial_nodes():
    spec_nodes = [
        {"id": "A", "kind": "symbol", "state": "pending"},
        {"id": "B", "kind": "symbol", "state": "pending"},
        {"id": "C", "kind": "symbol", "state": "pending"},
    ]

    g1 = build_graph(cast(GraphSpec, {"nodes": spec_nodes}))
    g2 = build_graph(cast(GraphSpec, {"nodes": list(reversed(spec_nodes))}))

    result1 = run_steps(g1, 3)
    result2 = run_steps(g2, 3)

    assert normalize_graph(result1) == normalize_graph(result2)


def test_prefix_monotonicity():
    base_spec = cast(GraphSpec, {"nodes": [{"id": "A", "kind": "symbol", "state": "pending"}]})
    base = build_graph(base_spec)

    g_n = run_steps(base, 2)
    g_n1 = run_steps(build_graph(base_spec), 3)

    norm_n = normalize_graph(g_n)
    norm_n1 = normalize_graph(g_n1)

    nodes_n = {n["id"] for n in norm_n["nodes"]}
    nodes_n1 = {n["id"] for n in norm_n1["nodes"]}
    edges_n = {(e["source"], e["target"], e["kind"]) for e in norm_n["edges"]}
    edges_n1 = {(e["source"], e["target"], e["kind"]) for e in norm_n1["edges"]}

    assert nodes_n.issubset(nodes_n1)
    assert edges_n.issubset(edges_n1)


def test_replay_stability():
    base_spec = cast(GraphSpec, {"nodes": [{"id": "A", "kind": "symbol", "state": "pending"}]})
    base = build_graph(base_spec)

    r1 = run_steps(base, 3)
    r2 = run_steps(build_graph(base_spec), 3)

    assert normalize_graph(r1) == normalize_graph(r2)


def test_edge_duplication_stability():
    base_nodes = [
        {"id": "A", "kind": "symbol", "state": "pending"},
        {"id": "B", "kind": "symbol", "state": "pending"},
    ]
    base_spec = cast(
        GraphSpec,
        {
            "nodes": base_nodes,
            "edges": [
                {"source": "A", "target": "B", "kind": "calls"},
            ],
        },
    )
    dup_spec = cast(
        GraphSpec,
        {
            "nodes": base_nodes,
            "edges": [
                {"source": "A", "target": "B", "kind": "calls"},
                {"source": "A", "target": "B", "kind": "calls"},
            ],
        },
    )

    g_base = build_graph(base_spec)
    g_dup = build_graph(dup_spec)

    result_base = run_steps(g_base, 3)
    result_dup = run_steps(g_dup, 3)

    assert normalize_graph(result_base) == normalize_graph(result_dup)
