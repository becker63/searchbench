from __future__ import annotations

import pytest

from harness.utils import repo_targets


def test_resolve_alias_small(monkeypatch, tmp_path):
    repo_dir = tmp_path / "small_repo"
    repo_dir.mkdir()
    monkeypatch.setenv("TEST_REPO_SMALL", str(repo_dir))
    assert repo_targets.resolve_repo_target("small") == str(repo_dir)


def test_resolve_alias_medium(monkeypatch, tmp_path):
    repo_dir = tmp_path / "medium_repo"
    repo_dir.mkdir()
    monkeypatch.setenv("TEST_REPO_MEDIUM", str(repo_dir))
    assert repo_targets.resolve_repo_target("medium") == str(repo_dir)


def test_resolve_alias_missing_env(monkeypatch):
    monkeypatch.delenv("TEST_REPO_SMALL", raising=False)
    with pytest.raises(ValueError):
        repo_targets.resolve_repo_target("small")


def test_resolve_path_passthrough(tmp_path):
    repo_dir = tmp_path / "path_repo"
    repo_dir.mkdir()
    assert repo_targets.resolve_repo_target(str(repo_dir)) == str(repo_dir)


def test_resolve_invalid_path_raises(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        repo_targets.resolve_repo_target(str(missing))


def test_unknown_target():
    with pytest.raises(ValueError):
        repo_targets.resolve_repo_target("unknown_target")
