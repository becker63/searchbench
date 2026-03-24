from __future__ import annotations

from pathlib import Path


def load_scoring_types(repo_root: Path) -> str:
    """
    Load canonical scoring type definitions from iterative-context.
    """
    types_path = repo_root / "iterative-context" / "src" / "iterative_context" / "types.py"
    if not types_path.exists():
        return ""
    try:
        return types_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_scoring_examples(repo_root: Path) -> str:
    scoring_path = repo_root / "iterative-context" / "src" / "iterative_context" / "scoring.py"
    if not scoring_path.exists():
        return ""
    try:
        return scoring_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def format_scoring_context(types: str, examples: str, max_chars: int = 8000) -> str:
    """
    Concatenate type definitions and reference implementations for prompt use.
    """
    canonical = "def score(node: object, state: object, context: object) -> float: ..."
    parts: list[str] = [f"# CANONICAL SIGNATURE\n{canonical}"]
    if types:
        parts.append(f"# TYPES (source of truth)\n{types}")
    if examples:
        parts.append(f"# EXAMPLES (reference implementations)\n{examples}")

    joined = "\n\n".join(parts)
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n\n# ... truncated"
    return joined
