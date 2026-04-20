from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def load_tests(repo_root: Path) -> List[Tuple[str, str]]:
    """
    Return a list of (relative_path, file_contents) for iterative-context tests.
    """
    tests_dir = repo_root / "iterative-context" / "tests"
    if not tests_dir.exists():
        return []

    files = sorted(tests_dir.rglob("*.py"))
    results: List[Tuple[str, str]] = []
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        rel_path = file_path.relative_to(repo_root)
        results.append((str(rel_path), content))

    return results


def format_tests_for_prompt(tests: List[Tuple[str, str]], max_chars: int = 12000) -> str:
    """
    Concatenate tests into a prompt-friendly string, truncating if necessary.
    """
    parts: list[str] = [f"# FILE: {path}\n{content}" for path, content in tests]
    joined = "\n\n".join(parts)
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n\n# ... truncated"
    return joined
