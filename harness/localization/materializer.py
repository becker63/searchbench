from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class RepoMaterializationRequest(BaseModel):
    """
    Input contract for repo materialization/resolution.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_config: str
    dataset_split: str
    repo_owner: str
    repo_name: str
    base_sha: str
    archive_uri: Optional[str] = None  # e.g., HF path or local cache key


class RepoMaterializationResult(BaseModel):
    """
    Output contract for repo materialization/resolution.
    """

    model_config = ConfigDict(extra="forbid")

    local_path: Optional[Path] = None
    cache_hit: Optional[bool] = None
    cache_validated: Optional[bool] = None
    mirror_fetched: Optional[bool] = None
    worktree_created: Optional[bool] = None
    checkout_verified: Optional[bool] = None
    logs: list[str] = []
    events: list[str] = []

    def describe_mapping(self) -> str:
        """
        Human-readable description of how the repo was mapped/materialized.
        """
        return (
            f"path={self.local_path}, cache_hit={self.cache_hit}, cache_validated={self.cache_validated}, "
            f"mirror_fetched={self.mirror_fetched}, worktree_created={self.worktree_created}, "
            f"checkout_verified={self.checkout_verified}"
        )

    def summary(self) -> str:
        """
        Concise human-readable summary for CLI/logs.
        """
        path = str(self.local_path) if self.local_path else "unset"
        status_parts = []
        if self.cache_hit is not None:
            status_parts.append(f"cache_hit={self.cache_hit}")
        if self.cache_validated is not None:
            status_parts.append(f"cache_validated={self.cache_validated}")
        if self.mirror_fetched is not None:
            status_parts.append(f"mirror_fetched={self.mirror_fetched}")
        if self.worktree_created is not None:
            status_parts.append(f"worktree_created={self.worktree_created}")
        if self.checkout_verified is not None:
            status_parts.append(f"checkout_verified={self.checkout_verified}")
        status = ", ".join(status_parts) if status_parts else "status=unknown"
        log_tail = f"; logs={self.logs[-1]}" if self.logs else ""
        return f"{path} ({status}){log_tail}"


@runtime_checkable
class RepoMaterializer(Protocol):
    """
    Abstract interface for materializing a repository at base_sha.
    Implementations are deferred to phase 2.
    """

    def materialize(self, request: RepoMaterializationRequest) -> RepoMaterializationResult:
        ...


def describe_repo_mapping(request: RepoMaterializationRequest) -> str:
    """
    Documented mapping rule: dataset/config/split + repo_owner/name + base_sha drive cache key and checkout target.
    Implementations SHOULD:
    - derive cache key from dataset/config/split + repo_owner/name + base_sha
    - download/extract archive if cache miss
    - checkout base_sha
    - return local path and status flags
    """
    return (
        f"cache_key={request.dataset_name}:{request.dataset_config}:{request.dataset_split}:"
        f"{request.repo_owner}/{request.repo_name}@{request.base_sha}"
    )
