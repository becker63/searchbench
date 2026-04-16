from __future__ import annotations

from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from .models import ComposeMode, ScoreResult


class ScoreBundle(BaseModel):
    """
    Complete output produced by `ScoreEngine` for one scored item.

    The bundle is deliberately separate from reusable batch summaries: it is
    part of the concrete scoring core and carries component results plus the
    composed machine/control score for one item.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str | None = None
    results: Mapping[str, ScoreResult]
    composed_score: float | None
    compose_mode: ComposeMode
    compose_weights: Mapping[str, float]
    available: bool
    diagnostics: Mapping[str, object] = Field(default_factory=dict)

    @property
    def component_scores(self) -> Mapping[str, float | None]:
        return {name: result.value for name, result in self.results.items()}

    @property
    def visible_component_scores(self) -> Mapping[str, float | None]:
        return {
            name: result.value
            for name, result in self.results.items()
            if result.visible_to_agent
        }

    @property
    def component_availability(self) -> Mapping[str, bool]:
        return {name: result.available for name, result in self.results.items()}


__all__ = ["ScoreBundle"]
