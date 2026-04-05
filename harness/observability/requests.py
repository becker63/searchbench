from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from harness.localization.models import LCAContext, LCAGold, LCATask, LCATaskIdentity

from .session_policy import SessionConfig


class LocalizationAdHocRequest(BaseModel):
    """Public request model for ad hoc localization runs (CLI boundary)."""

    model_config = ConfigDict(extra="forbid")

    identity: LCATaskIdentity
    context: LCAContext
    gold: LCAGold
    repo: str | None = None
    session: SessionConfig | None = None

    def to_task(self) -> LCATask:
        return LCATask(identity=self.identity, context=self.context, gold=self.gold, repo=self.repo)


class HostedLocalizationBaselineRequest(BaseModel):
    """Public request model for hosted localization baseline experiments."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    version: str | None = None
    dataset_config: str | None = None
    dataset_split: str | None = None
    session: SessionConfig | None = None


class HostedLocalizationRunRequest(BaseModel):
    """Public request model for hosted localization experiments."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    version: str | None = None
    dataset_config: str | None = None
    dataset_split: str | None = None
    session: SessionConfig | None = None
