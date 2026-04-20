from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

_OPTIMIZATION_REPO_ROOT: ContextVar[Path | None] = ContextVar(
    "optimization_repo_root",
    default=None,
)
_OPTIMIZATION_POLICY_PATH: ContextVar[Path | None] = ContextVar(
    "optimization_policy_path",
    default=None,
)


def optimization_repo_root() -> Path | None:
    return _OPTIMIZATION_REPO_ROOT.get()


def optimization_policy_path(default: Path) -> Path:
    return _OPTIMIZATION_POLICY_PATH.get() or default


@contextmanager
def isolated_optimization_workspace(repo_root: Path) -> Iterator[None]:
    repo_root = repo_root.resolve()
    policy_path = repo_root / "harness" / "policy" / "current.py"
    repo_token = _OPTIMIZATION_REPO_ROOT.set(repo_root)
    policy_token = _OPTIMIZATION_POLICY_PATH.set(policy_path)
    try:
        yield
    finally:
        _OPTIMIZATION_POLICY_PATH.reset(policy_token)
        _OPTIMIZATION_REPO_ROOT.reset(repo_token)

