from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Optional

from .materializer import RepoMaterializationRequest, RepoMaterializationResult, RepoMaterializer, describe_repo_mapping


class HuggingFaceRepoMaterializer(RepoMaterializer):
    """
    Repo materializer for HF-backed LCA tasks using a cached bare git mirror + per-task worktree.
    Concurrency is enforced per cache key via a lock file; partial builds are never reused.
    """

    def __init__(
        self,
        cache_root: Path | None = None,
        repo_base_url: str | None = None,
        lock_timeout_seconds: int = 120,
        stale_lock_timeout_seconds: int = 300,
    ) -> None:
        self.cache_root = cache_root or Path(os.environ.get("HF_LCA_CACHE_DIR", Path.home() / ".cache" / "harness" / "hf_repos"))
        self.repo_base_url = repo_base_url or os.environ.get("HF_LCA_REPO_BASE", "https://github.com")
        self.lock_timeout_seconds = lock_timeout_seconds
        self.stale_lock_timeout_seconds = stale_lock_timeout_seconds

    def materialize(self, request: RepoMaterializationRequest) -> RepoMaterializationResult:
        cache_key = describe_repo_mapping(request).replace(":", "__").replace("/", "__")
        worktree_root = self.cache_root / "worktrees" / cache_key
        mirror_root = self.cache_root / "mirrors"
        lock_path = self.cache_root / "locks" / f"{cache_key}.lock"
        completion_marker = worktree_root / ".complete"
        repo_url = f"{self.repo_base_url.rstrip('/')}/{request.repo_owner}/{request.repo_name}.git"
        mirror_path = mirror_root / f"{request.repo_owner}__{request.repo_name}.git"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        mirror_root.mkdir(parents=True, exist_ok=True)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        logs: List[str] = []
        mirror_fetched = False
        worktree_created = False
        checkout_verified = False

        with _Lock(
            lock_path,
            timeout_seconds=self.lock_timeout_seconds,
            stale_timeout_seconds=self.stale_lock_timeout_seconds,
            logs=logs,
        ):
            if worktree_root.exists() and not completion_marker.exists():
                # Incomplete prior build: remove and rebuild
                logs.append(f"incomplete cache detected at {worktree_root}; pruning and rebuilding")
                shutil.rmtree(worktree_root, ignore_errors=True)
                _prune_worktree(mirror_path, worktree_root, logs)
            if completion_marker.exists() and not worktree_root.exists():
                logs.append(f"cache marker present without worktree at {completion_marker}; rebuilding")
                completion_marker.unlink(missing_ok=True)
            if completion_marker.exists() and worktree_root.exists():
                try:
                    _validate_worktree(worktree_root, request.base_sha, logs)
                    cache_logs = [f"reused cached worktree at {worktree_root}"]
                    return RepoMaterializationResult(
                        local_path=worktree_root,
                        cache_hit=True,
                        cache_validated=True,
                        mirror_fetched=False,
                        worktree_created=False,
                        checkout_verified=True,
                        logs=cache_logs,
                        events=cache_logs,
                    )
                except Exception as exc:
                    logs.append(f"cached worktree invalid: {exc}; pruning and rebuilding")
                    shutil.rmtree(worktree_root, ignore_errors=True)
                    _prune_worktree(mirror_path, worktree_root, logs)
                    completion_marker.unlink(missing_ok=True)

            try:
                mirror_fetched = _ensure_mirror_has_commit(mirror_path, repo_url, request.base_sha, logs)
            except Exception as exc:
                logs.append(f"failed to fetch {request.base_sha} from {repo_url}: {exc}")
                raise MaterializationError(f"fetch failed: {exc}", category="fetch") from exc

            _prune_worktree(mirror_path, worktree_root, logs)

            try:
                if worktree_root.exists():
                    shutil.rmtree(worktree_root, ignore_errors=True)
                worktree_root.parent.mkdir(parents=True, exist_ok=True)
                _create_worktree(mirror_path, worktree_root, request.base_sha, logs)
                worktree_created = True
                _verify_checkout(worktree_root, request.base_sha, logs)
                checkout_verified = True
                completion_marker.touch(exist_ok=True)
            except Exception:
                shutil.rmtree(worktree_root, ignore_errors=True)
                _prune_worktree(mirror_path, worktree_root, logs)
                raise

        return RepoMaterializationResult(
            local_path=worktree_root,
            cache_hit=False,
            cache_validated=False,
            mirror_fetched=mirror_fetched,
            worktree_created=worktree_created,
            checkout_verified=checkout_verified,
            logs=logs,
            events=logs,
        )


class MaterializationError(RuntimeError):
    def __init__(self, message: str, category: str = "materialization"):
        super().__init__(message)
        self.category = category


class _Lock:
    """
    Simple per-key file lock using exclusive file creation and retry.
    """

    def __init__(
        self,
        path: Path,
        timeout_seconds: int = 120,
        poll_interval: float = 0.25,
        stale_timeout_seconds: int = 300,
        logs: Optional[List[str]] = None,
    ) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.stale_timeout_seconds = stale_timeout_seconds
        self.logs = logs if logs is not None else []
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        deadline = time.time() + self.timeout_seconds
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, f"{os.getpid()}:{int(time.time())}".encode("utf-8"))
                return
            except FileExistsError:
                if self._maybe_break_stale_lock():
                    continue
                if time.time() > deadline:
                    raise MaterializationError(f"Timed out acquiring lock {self.path}", category="lock")
                time.sleep(self.poll_interval)

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass

    def _maybe_break_stale_lock(self) -> bool:
        try:
            data = self.path.read_text().strip()
            pid_str, ts_str = data.split(":")
            pid = int(pid_str)
            ts = int(ts_str)
        except Exception:
            pid = None
            ts = None
        now = time.time()
        is_stale = False
        if ts is not None and now - ts > self.stale_timeout_seconds:
            is_stale = True
            self.logs.append(f"stale lock detected at {self.path}, age={now - ts:.1f}s; breaking")
        elif pid is not None and not _pid_alive(pid):
            is_stale = True
            self.logs.append(f"stale lock pid={pid} not alive at {self.path}; breaking")
        if is_stale:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            return True
        return False

    def __enter__(self) -> "_Lock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _run_git(args: Iterable[str], cwd: Path | None = None, logs: List[str] | None = None) -> str:
    import subprocess

    cmd = ["git", *args]
    completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    out = completed.stdout.strip()
    if logs is not None:
        logs.append(f"git {' '.join(args)} -> {out}")
    return out


def _ensure_mirror_has_commit(mirror_path: Path, repo_url: str, base_sha: str, logs: List[str]) -> bool:
    if not mirror_path.exists():
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["init", "--bare", str(mirror_path)], logs=logs)
        _run_git(["--git-dir", str(mirror_path), "remote", "add", "origin", repo_url], logs=logs)
    try:
        _run_git(["--git-dir", str(mirror_path), "cat-file", "-e", base_sha], logs=None)
    except Exception:
        _run_git(["--git-dir", str(mirror_path), "fetch", "--depth=1", "origin", base_sha], logs=logs)
        return True
    return False


def _create_worktree(mirror_path: Path, worktree_path: Path, base_sha: str, logs: List[str]) -> None:
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["--git-dir", str(mirror_path), "worktree", "add", str(worktree_path), base_sha], logs=logs)


def _verify_checkout(worktree_path: Path, base_sha: str, logs: List[str]) -> None:
    head = _run_git(["-C", str(worktree_path), "rev-parse", "HEAD"], logs=logs)
    if head.strip() != base_sha:
        raise RuntimeError(f"Worktree HEAD {head} does not match expected base_sha {base_sha}")


def _validate_worktree(worktree_path: Path, base_sha: str, logs: List[str]) -> None:
    if not worktree_path.exists():
        raise RuntimeError("worktree path missing")
    _verify_checkout(worktree_path, base_sha, logs)


def _prune_worktree(mirror_path: Path, worktree_path: Path, logs: List[str]) -> None:
    """
    Ensure git worktree metadata is pruned for removed paths to avoid reuse errors.
    """
    if mirror_path.exists():
        try:
            _run_git(["--git-dir", str(mirror_path), "worktree", "prune"], logs=logs)
        except Exception:
            # pruning is best-effort; continue
            pass
    if worktree_path.exists() and not (worktree_path / ".complete").exists():
        shutil.rmtree(worktree_path, ignore_errors=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True
    return True
