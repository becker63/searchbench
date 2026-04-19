from pathlib import Path

import pytest

from iterative_context.test_helpers.repos import get_repo


@pytest.fixture(params=["small", "medium"])
def repo(request: pytest.FixtureRequest) -> tuple[str, Path]:
    name = request.param
    return name, get_repo(name)


def test_repo_fixture_exists(repo: tuple[str, Path]) -> None:
    """Ensure fixture repos are wired correctly and exist on disk."""
    name, path = repo

    assert path is not None, f"{name} repo path is None"
    assert path.exists(), f"{name} repo path does not exist: {path}"
    assert path.is_dir(), f"{name} repo path is not a directory: {path}"


def test_repo_has_pyproject(repo: tuple[str, Path]) -> None:
    """Ensure repos look like real Python projects (minimal sanity check)."""
    name, path = repo

    pyproject = path / "pyproject.toml"

    assert pyproject.exists(), (
        f"{name} repo missing pyproject.toml — not a valid test target: {path}"
    )
