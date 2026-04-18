from __future__ import annotations
import concurrent.futures
import shutil
import subprocess
from pathlib import Path

from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity
from harness.localization.materialization.worktree import (
    RepoMaterializationRequest,
    RepoMaterializationResult,
    WorktreeManager,
)


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr}")
    return completed.stdout.strip()


def _make_local_bare_repo(tmp_path: Path) -> tuple[Path, str]:
    owner = "example"
    repo_name = "repo"
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    repo_path = workdir / repo_name
    repo_path.mkdir()
    _run_git(["init"], cwd=repo_path)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo_path)
    _run_git(["config", "user.name", "Test User"], cwd=repo_path)
    (repo_path / "README.md").write_text("hello\n")
    _run_git(["add", "README.md"], cwd=repo_path)
    _run_git(["commit", "-m", "init"], cwd=repo_path)
    base_sha = _run_git(["rev-parse", "HEAD"], cwd=repo_path)

    origin_root = tmp_path / "origin"
    origin_repo_dir = origin_root / owner
    origin_repo_dir.mkdir(parents=True, exist_ok=True)
    origin_bare = origin_repo_dir / f"{repo_name}.git"
    _run_git(["clone", "--bare", str(repo_path), str(origin_bare)])
    return origin_root, base_sha


def _task(repo_owner: str, repo_name: str, base_sha: str, config: str = "py") -> LCATask:
    identity = LCATaskIdentity(
        dataset_name="hf",
        dataset_config=config,
        dataset_split="dev",
        repo_owner=repo_owner,
        repo_name=repo_name,
        base_sha=base_sha,
    )
    return LCATask(
        identity=identity,
        context=LCAContext(issue_title="t", issue_body="b"),
        gold=LCAGold(changed_files=["README.md"]),
    )


def _worktree_path(manager: WorktreeManager, request: RepoMaterializationRequest) -> Path:
    return manager._worktree_path(manager._cache_key(request))  # pyright: ignore[reportPrivateUsage]


def test_worktree_manager_creates_and_reuses_cache(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    manager = WorktreeManager(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )

    first = manager.get_or_create(request)
    assert first.local_path and first.local_path.exists()
    assert first.cache_hit is False
    assert first.worktree_created is True
    assert first.checkout_verified is True
    assert first.mirror_fetched is True
    assert (first.local_path / ".complete").exists()

    second = manager.get_or_create(request)
    assert second.cache_hit is True
    assert second.local_path == first.local_path
    assert second.checkout_verified is True
    assert second.worktree_created is False
    assert second.cache_validated is True
    assert second.mirror_fetched is False


def test_worktree_manager_rebuilds_incomplete(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    manager = WorktreeManager(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )

    result = manager.get_or_create(request)
    marker = result.local_path / ".complete" if result.local_path else None
    if marker and marker.exists():
        marker.unlink()
    # simulate partial build by removing marker and wiping contents
    if result.local_path:
        shutil.rmtree(result.local_path / ".git", ignore_errors=True)

    rebuilt = manager.get_or_create(request)
    assert rebuilt.local_path and rebuilt.local_path.exists()
    assert rebuilt.cache_hit is False  # incomplete cache forces rebuild
    assert (rebuilt.local_path / ".complete").exists()
    assert rebuilt.checkout_verified is True
    assert rebuilt.cache_validated is False


def test_worktree_manager_reuses_after_multiple_runs(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    manager = WorktreeManager(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )
    first = manager.get_or_create(request)
    second = manager.get_or_create(request)
    third = manager.get_or_create(request)
    assert first.local_path == second.local_path == third.local_path
    assert [first.cache_hit, second.cache_hit, third.cache_hit] == [False, True, True]
    assert third.checkout_verified is True
    assert third.cache_validated is True


def test_worktree_manager_cache_identity_ignores_dataset_metadata(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    manager = WorktreeManager(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )
    same_checkout_request = request.model_copy(
        update={
            "dataset_name": "other-dataset",
            "dataset_config": "java",
            "dataset_split": "test",
        }
    )

    first = manager.get_or_create(request)
    second = manager.get_or_create(same_checkout_request)

    assert second.cache_hit is True
    assert second.local_path == first.local_path


def test_worktree_manager_cache_identity_changes_with_sha(tmp_path):
    manager = WorktreeManager(cache_root=tmp_path / "cache")
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha="a" * 40,
    )
    other_sha_request = request.model_copy(update={"base_sha": "b" * 40})

    assert _worktree_path(manager, request) != _worktree_path(manager, other_sha_request)


def test_stale_lock_is_broken(tmp_path, monkeypatch):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    manager = WorktreeManager(
        cache_root=tmp_path / "cache",
        repo_base_url=origin_root.as_uri(),
        lock_timeout_seconds=2,
        stale_lock_timeout_seconds=0,  # break immediately
    )
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )
    lock_path = manager._lock_path(manager._cache_key(request))  # pyright: ignore[reportPrivateUsage]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("99999:0")

    result = manager.get_or_create(request)
    assert result.cache_hit is False
    assert any("stale lock" in log for log in result.logs)
    assert result.cache_validated is False


def test_invalid_cached_head_triggers_rebuild(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    manager = WorktreeManager(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )
    first = manager.get_or_create(request)
    assert first.cache_hit is False
    # mutate worktree HEAD to wrong commit by creating a new commit
    assert first.local_path
    (first.local_path / "mutate.txt").write_text("mutate\n")
    _run_git(["add", "mutate.txt"], cwd=first.local_path)
    _run_git(["commit", "-m", "mutate"], cwd=first.local_path)

    second = manager.get_or_create(request)
    assert second.cache_hit is False  # cache invalid -> rebuild
    assert second.checkout_verified is True
    assert second.cache_validated is False


def test_checkout_summary_mentions_git_states(tmp_path):
    dummy = RepoMaterializationResult(
        local_path=tmp_path,
        cache_hit=True,
        mirror_fetched=False,
        worktree_created=True,
        checkout_verified=True,
        logs=["log entry"],
    )
    summary_text = dummy.summary()
    assert "mirror_fetched" in summary_text
    assert "worktree_created" in summary_text
    assert "checkout_verified" in summary_text


def test_worktree_manager_parallel_calls_share_cache(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    manager = WorktreeManager(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )

    def _get_or_create():
        return manager.get_or_create(request)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: _get_or_create(), range(2)))

    assert len(results) == 2
    assert results[0].local_path == results[1].local_path
    assert (results[0].local_path / ".complete").exists()


def test_worktree_manager_targeted_invalidation_only_removes_selected_worktree(tmp_path):
    manager = WorktreeManager(cache_root=tmp_path / "cache")
    selected_request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha="a" * 40,
    )
    other_request = selected_request.model_copy(
        update={"dataset_config": "java", "base_sha": "b" * 40}
    )
    selected_path = _worktree_path(manager, selected_request)
    other_path = _worktree_path(manager, other_request)
    selected_path.mkdir(parents=True)
    other_path.mkdir(parents=True)
    (selected_path / ".complete").write_text("")
    (other_path / ".complete").write_text("")

    events = manager.invalidate_tasks([
        _task("example", "repo", "a" * 40, config="py")
    ])

    assert any("removed cached worktree" in event for event in events)
    assert not selected_path.exists()
    assert other_path.exists()
    assert (other_path / ".complete").exists()


def test_worktree_manager_invalidation_forces_next_materialization_fresh(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    manager = WorktreeManager(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )

    first = manager.get_or_create(request)
    assert first.cache_hit is False
    assert first.worktree_created is True
    second = manager.get_or_create(request)
    assert second.cache_hit is True
    assert second.worktree_created is False

    manager.invalidate(request)
    fresh = manager.get_or_create(request)

    assert fresh.cache_hit is False
    assert fresh.worktree_created is True
    assert fresh.local_path == first.local_path
    assert fresh.local_path and (fresh.local_path / ".complete").exists()


def test_worktree_manager_invalidation_is_idempotent_for_missing_or_corrupt_worktree(tmp_path):
    manager = WorktreeManager(cache_root=tmp_path / "cache")
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha="a" * 40,
    )
    worktree_path = _worktree_path(manager, request)
    worktree_path.mkdir(parents=True)

    first_events = manager.invalidate(request)
    second_events = manager.invalidate(request)

    assert not worktree_path.exists()
    assert any("removed cached worktree" in event for event in first_events)
    assert any("no cached worktree found" in event for event in second_events)
