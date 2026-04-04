from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

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
    downloaded: Optional[bool] = None
    extracted: Optional[bool] = None
    checked_out: Optional[bool] = None
    logs: list[str] = []

    def describe_mapping(self) -> str:
        """
        Human-readable description of how the repo was mapped/materialized.
        """
        return (
            f"path={self.local_path}, cache_hit={self.cache_hit}, "
            f"downloaded={self.downloaded}, extracted={self.extracted}, checked_out={self.checked_out}"
        )


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
