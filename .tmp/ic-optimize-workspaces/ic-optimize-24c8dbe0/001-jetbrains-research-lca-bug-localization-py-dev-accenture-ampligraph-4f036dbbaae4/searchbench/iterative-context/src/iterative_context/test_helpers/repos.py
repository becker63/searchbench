# src/iterative_context/test_helpers/repos.py

import os
from collections.abc import Iterator
from pathlib import Path


def get_repo(name: str) -> Path:
    env_var = f"TEST_REPO_{name.upper()}"
    path = os.environ.get(env_var)

    if not path:
        raise RuntimeError(f"{env_var} not set")

    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"{env_var} path does not exist: {p}")

    return p


def iter_repos() -> Iterator[tuple[str, Path]]:
    for name in ["small", "medium"]:
        yield name, get_repo(name)
