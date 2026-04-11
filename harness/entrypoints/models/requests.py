from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator

from harness.telemetry.tracing.session_policy import SessionConfig


class LocalizationDatasetSource(str, Enum):
    LANGFUSE = "langfuse"
    HUGGINGFACE = "huggingface"


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
    max_workers: int = 1
    selection: dict[str, object] | None = None
    projection: dict[str, object] | None = None

    @field_validator("max_workers")
    @classmethod
    def _validate_workers(cls, value: int) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("max_workers must be a positive integer")
        return value


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
    max_workers: int = 1
    selection: dict[str, object] | None = None
    projection: dict[str, object] | None = None

    @field_validator("max_workers")
    @classmethod
    def _validate_workers(cls, value: int) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("max_workers must be a positive integer")
        return value
