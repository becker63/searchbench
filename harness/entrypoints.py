from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .observability.baselines import BaselineBundle
from .observability.session_policy import SessionConfig


class AdHocOptimizationRequest(BaseModel):
    """Public request model for ad hoc optimization runs (CLI boundary)."""

    model_config = ConfigDict(extra="forbid")

    repo_ref: str
    symbol: str
    iterations: int = 3
    session: SessionConfig | None = None


class HostedJCBaselineRequest(BaseModel):
    """Public request model for hosted JC baseline experiments."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    version: str | None = None
    session: SessionConfig | None = None


class HostedICOptimizationRequest(BaseModel):
    """Public request model for hosted IC optimization experiments."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    version: str | None = None
    iterations: int = 5
    baselines: BaselineBundle | None = None
    recompute_baselines: bool = False
    session: SessionConfig | None = None
