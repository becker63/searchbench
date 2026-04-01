from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field


class TestSpan(BaseModel):
    """Pydantic test span with scoring support for Langfuse stubs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    trace_id: str | None = None
    scores: list[tuple[str, float, dict[str, object] | None]] = Field(default_factory=list)

    def score(self, name: str, value: float | int | bool, metadata: dict[str, object] | None = None) -> None:
        self.scores.append((name, float(value), metadata))

    def end(self, **kwargs: Any) -> None:  # pragma: no cover - simple stub
        return None


@pytest.fixture
def test_span() -> TestSpan:
    """Reusable span fixture with scoring and trace id."""
    return TestSpan(id="span", trace_id="trace")
