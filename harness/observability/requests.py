from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .session_policy import SessionConfig


class HostedLocalizationBaselineRequest(BaseModel):
    """Public request model for hosted localization baseline experiments."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    version: str | None = None
    dataset_config: str | None = None
    dataset_split: str | None = None
    session: SessionConfig | None = None
    max_items: int | None = None
    offset: int = 0
    selection: dict[str, object] | None = None
    projection: dict[str, object] | None = None


class HostedLocalizationRunRequest(BaseModel):
    """Public request model for hosted localization experiments."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    version: str | None = None
    dataset_config: str | None = None
    dataset_split: str | None = None
    session: SessionConfig | None = None
    max_items: int | None = None
    offset: int = 0
    selection: dict[str, object] | None = None
    projection: dict[str, object] | None = None
