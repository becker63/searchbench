from __future__ import annotations
import concurrent.futures
import shutil
import subprocess
from pathlib import Path

from harness.localization.materialization.hf_materialize import HuggingFaceRepoMaterializer
from harness.localization.materialization.materialize import RepoMaterializationRequest, RepoMaterializationResult, describe_repo_mapping


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


def test_materializer_creates_and_reuses_cache(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    materializer = HuggingFaceRepoMaterializer(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )

    first = materializer.materialize(request)
    assert first.local_path and first.local_path.exists()
    assert first.cache_hit is False
    assert first.worktree_created is True
    assert first.checkout_verified is True
    assert first.mirror_fetched is True
    assert (first.local_path / ".complete").exists()

    second = materializer.materialize(request)
    assert second.cache_hit is True
    assert second.local_path == first.local_path
    assert second.checkout_verified is True
    assert second.worktree_created is False
    assert second.cache_validated is True
    assert second.mirror_fetched is False


def test_materializer_rebuilds_incomplete(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    materializer = HuggingFaceRepoMaterializer(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )

    result = materializer.materialize(request)
    marker = result.local_path / ".complete" if result.local_path else None
    if marker and marker.exists():
        marker.unlink()
    # simulate partial build by removing marker and wiping contents
    if result.local_path:
        shutil.rmtree(result.local_path / ".git", ignore_errors=True)

    rebuilt = materializer.materialize(request)
    assert rebuilt.local_path and rebuilt.local_path.exists()
    assert rebuilt.cache_hit is False  # incomplete cache forces rebuild
    assert (rebuilt.local_path / ".complete").exists()
    assert rebuilt.checkout_verified is True
    assert rebuilt.cache_validated is False


def test_materializer_reuses_after_multiple_runs(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    materializer = HuggingFaceRepoMaterializer(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )
    first = materializer.materialize(request)
    second = materializer.materialize(request)
    third = materializer.materialize(request)
    assert first.local_path == second.local_path == third.local_path
    assert [first.cache_hit, second.cache_hit, third.cache_hit] == [False, True, True]
    assert third.checkout_verified is True
    assert third.cache_validated is True


def test_stale_lock_is_broken(tmp_path, monkeypatch):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    materializer = HuggingFaceRepoMaterializer(
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
    cache_key = describe_repo_mapping(request).replace(":", "__").replace("/", "__")
    lock_path = materializer.cache_root / "locks" / f"{cache_key}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("99999:0")

    result = materializer.materialize(request)
    assert result.cache_hit is False
    assert any("stale lock" in log for log in result.logs)
    assert result.cache_validated is False


def test_invalid_cached_head_triggers_rebuild(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    materializer = HuggingFaceRepoMaterializer(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )
    first = materializer.materialize(request)
    assert first.cache_hit is False
    # mutate worktree HEAD to wrong commit by creating a new commit
    assert first.local_path
    (first.local_path / "mutate.txt").write_text("mutate\n")
    _run_git(["add", "mutate.txt"], cwd=first.local_path)
    _run_git(["commit", "-m", "mutate"], cwd=first.local_path)

    second = materializer.materialize(request)
    assert second.cache_hit is False  # cache invalid -> rebuild
    assert second.checkout_verified is True
    assert second.cache_validated is False


def test_materialization_summary_mentions_git_states(tmp_path):
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


def test_materializer_parallel_calls_share_cache(tmp_path):
    origin_root, base_sha = _make_local_bare_repo(tmp_path)
    materializer = HuggingFaceRepoMaterializer(cache_root=tmp_path / "cache", repo_base_url=origin_root.as_uri())
    request = RepoMaterializationRequest(
        dataset_name="hf",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="example",
        repo_name="repo",
        base_sha=base_sha,
    )

    def _materialize():
        return materializer.materialize(request)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: _materialize(), range(2)))

    assert len(results) == 2
    assert results[0].local_path == results[1].local_path
    assert (results[0].local_path / ".complete").exists()
