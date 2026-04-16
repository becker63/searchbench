from __future__ import annotations

from typing import Mapping

from .bundle import ScoreBundle
from .helpers import clamp_01
from .models import ComposeMode, ScoreComponent, ScoreConfig, ScoreContext, ScoreResult
from .scorers import SCORE_REGISTRY


class ScoreEngine:
    def __init__(
        self,
        registry: Mapping[str, ScoreComponent] | None = None,
        config: ScoreConfig | None = None,
    ) -> None:
        self._registry = dict(registry or SCORE_REGISTRY)
        self._config = config or ScoreConfig.default()
        self._validate_registry()

    def _validate_registry(self) -> None:
        missing_enabled = [name for name in self._config.enabled if name not in self._registry]
        if missing_enabled:
            raise KeyError(f"Unknown enabled scores: {missing_enabled}")
        missing_composed = [
            name for name in self._config.compose_weights if name not in self._registry
        ]
        if missing_composed:
            raise KeyError(f"Unknown composed scores: {missing_composed}")

    def evaluate(self, ctx: ScoreContext) -> ScoreBundle:
        results: dict[str, ScoreResult] = {}
        visible_names = set(self._config.visible_to_agent)
        for name in self._config.enabled:
            raw = self._registry[name].compute(ctx)
            results[name] = raw.model_copy(update={"visible_to_agent": name in visible_names})
        composed_score = self._compose(results)
        return ScoreBundle(
            results=results,
            composed_score=composed_score,
            compose_mode=self._config.compose_mode,
            compose_weights=dict(self._config.compose_weights),
            available=composed_score is not None,
        )

    def _compose(self, results: Mapping[str, ScoreResult]) -> float | None:
        selected: list[tuple[float, float]] = []
        for name, weight in self._config.compose_weights.items():
            result = results[name]
            if not result.available or result.value is None:
                return None
            selected.append((clamp_01(result.value), float(weight)))

        total_weight = sum(weight for _, weight in selected)
        normalized = [(value, weight / total_weight) for value, weight in selected]
        if self._config.compose_mode == ComposeMode.WEIGHTED_SUM:
            return sum(value * weight for value, weight in normalized)
        if self._config.compose_mode == ComposeMode.GEOMETRIC_MEAN:
            product = 1.0
            for value, weight in normalized:
                product *= max(self._config.epsilon, value) ** weight
            return product
        raise AssertionError(f"Unhandled compose mode: {self._config.compose_mode}")


__all__ = ["ScoreEngine"]
