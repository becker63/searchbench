from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ScorerContext(BaseModel):
    """Structured scorer context for writer prompts."""

    model_config = ConfigDict(extra="forbid")

    signature: str
    graph_models: str
    types: str
    examples: str
    notes: str = ""


def load_graph_models(repo_root: Path) -> str:
    """
    Load graph model definitions for scorer context.
    """
    models_path = (
        repo_root
        / "iterative-context"
        / "src"
        / "iterative_context"
        / "graph_models.py"
    )
    if not models_path.exists():
        return ""
    try:
        return models_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_scoring_types(repo_root: Path) -> str:
    """
    Load canonical scoring type definitions from iterative-context.
    """
    types_path = (
        repo_root / "iterative-context" / "src" / "iterative_context" / "types.py"
    )
    if not types_path.exists():
        return ""
    try:
        return types_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_scoring_examples(repo_root: Path) -> str:
    scoring_paths = [
        repo_root / "harness" / "scoring_examples.py",
        repo_root / "scoring_examples.py",
    ]
    for scoring_path in scoring_paths:
        if not scoring_path.exists():
            continue
        try:
            return scoring_path.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


def build_scorer_context(repo_root: Path) -> ScorerContext:
    signature = "SelectionCallable = Callable[[GraphNode, Graph, int], float]"
    return ScorerContext(
        signature=signature,
        graph_models=load_graph_models(repo_root),
        types=load_scoring_types(repo_root),
        examples=load_scoring_examples(repo_root),
        notes="Traversal calls score(node, graph, step) deterministically; use GraphNode/Graph shapes above.",
    )


def format_scoring_context(ctx: ScorerContext, max_chars: int = 8000) -> str:
    """
    Concatenate type definitions and reference implementations for prompt use.
    """
    parts: list[str] = [f"# CANONICAL SIGNATURE\n{ctx.signature}"]
    if ctx.graph_models:
        parts.append(
            "# GRAPH MODELS (Graph, GraphNode, Node variants, Edge/Event types)\n"
            + ctx.graph_models
        )
    if ctx.types:
        parts.append(f"# TYPES (source of truth)\n{ctx.types}")
    if ctx.examples:
        parts.append(f"# EXAMPLES (reference implementations)\n{ctx.examples}")
    if ctx.notes:
        parts.append(f"# NOTES\n{ctx.notes}")

    joined = "\n\n".join(parts)
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n\n# ... truncated"
    return joined
