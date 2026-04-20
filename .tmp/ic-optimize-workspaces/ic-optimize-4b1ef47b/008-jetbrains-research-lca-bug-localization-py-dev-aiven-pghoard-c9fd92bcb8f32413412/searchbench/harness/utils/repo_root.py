from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Walk upward until the monorepo root is found.

    Preference order:
    1) A directory named "searchbench"
    2) A directory containing both pyproject.toml and uv.lock
    """
    current = start or Path.cwd()
    current = current.resolve()

    for path in [current, *current.parents]:
        if path.name == "searchbench":
            return path
        if (path / "pyproject.toml").exists() and (path / "uv.lock").exists():
            return path
    raise FileNotFoundError("Repository root not found (missing searchbench dir or pyproject.toml and uv.lock)")
