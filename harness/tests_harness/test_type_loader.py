from __future__ import annotations

from harness.utils import type_loader


def test_format_scoring_context_includes_selection_and_graph_models():
    ctx = type_loader.ScorerContext(
        signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
        graph_models="class GraphNode:\n    pass\nclass Graph:\n    ...\n",
        types="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
        examples="def score(node, graph, step): ...",
        notes="Traversal uses score(node, graph, step).",
    )
    rendered = type_loader.format_scoring_context(ctx)
    assert "SelectionCallable" in rendered
    assert "GraphNode" in rendered
    assert "Graph" in rendered


def test_load_scoring_examples_uses_local_file(tmp_path):
    examples = (
        "from iterative_context.graph_models import Graph, GraphNode\n"
        "def score(node: GraphNode, graph: Graph, step: int) -> float:\n"
        "    return 1.0\n"
    )
    (tmp_path / "scoring_examples.py").write_text(examples, encoding="utf-8")
    loaded = type_loader.load_scoring_examples(tmp_path)
    assert "GraphNode" in loaded
    assert "score(node: GraphNode, graph: Graph, step: int)" in loaded
