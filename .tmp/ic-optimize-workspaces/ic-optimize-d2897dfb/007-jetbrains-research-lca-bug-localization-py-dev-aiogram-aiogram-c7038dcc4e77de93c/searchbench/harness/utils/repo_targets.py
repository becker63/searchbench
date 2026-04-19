from __future__ import annotations

from pathlib import Path
from typing import Dict

from .env import getenv


_ALIASES = {
    "small": "TEST_REPO_SMALL",
    "medium": "TEST_REPO_MEDIUM",
    "itsdangerous": "TEST_REPO_SMALL",
    "click": "TEST_REPO_MEDIUM",
}


def list_repo_targets() -> Dict[str, str]:
    return {alias: var for alias, var in _ALIASES.items()}


def is_known_repo_target(repo_ref: str) -> bool:
    return repo_ref in _ALIASES


def _resolve_env_var(var: str) -> str:
    val = getenv(var)
    if not val:
        raise ValueError(f"Environment variable {var} not set for repo target")
    return val


def resolve_repo_target(repo_ref: str) -> str:
    # Alias -> env var
    if repo_ref in _ALIASES:
        path = _resolve_env_var(_ALIASES[repo_ref])
        if not Path(path).exists():
            raise FileNotFoundError(f"Resolved repo path does not exist for {repo_ref}: {path}")
        return path

    candidate = Path(repo_ref)
    if candidate.exists():
        return str(candidate)

    # Looks like a path but missing
    if any(sep in repo_ref for sep in ("/", "\\")) or repo_ref.startswith("."):
        raise FileNotFoundError(f"Repo path does not exist: {repo_ref}")

    raise ValueError(f"Unknown repo target: {repo_ref}")


__all__ = ["resolve_repo_target", "list_repo_targets", "is_known_repo_target"]
