from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import log1p
from typing import Protocol

# ---------------------------------
# Basic evaluator inputs
# ---------------------------------


@dataclass(frozen=True)
class LocalizationMetrics:
    precision: float
    recall: float
    f1: float
    hit: float
    score: float  # legacy benchmark score (kept unchanged)


@dataclass(frozen=True)
class ScoreContext:
    """
    Shared input for all score plugins.

    In the real harness, this would be built after:
    - prediction is produced
    - gold files are known (evaluation side)
    - graph distances are computed
    - token usage is extracted
    """

    metrics: LocalizationMetrics
    gold_min_hops: int | None = None
    issue_min_hops: int | None = None
    previous_total_tokens: int | None = None


# ---------------------------------
# Score result model
# ---------------------------------


class Goal(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class ScoreResult:
    name: str
    value: float | None
    goal: Goal
    visible_to_agent: bool
    available: bool
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------
# Plugin interface
# ---------------------------------


class ScoreComponent(Protocol):
    name: str
    default_enabled: bool
    default_visible_to_agent: bool

    def compute(self, ctx: ScoreContext) -> ScoreResult: ...


# ---------------------------------
# Helpers
# ---------------------------------


def hop_to_score(hops: int | None) -> float | None:
    """
    Smaller hop distance is better.
      0 -> 1.0
      1 -> 0.5
      2 -> 0.333...
    """
    if hops is None:
        return None
    if hops < 0:
        raise ValueError("hops must be >= 0")
    return 1.0 / (1.0 + float(hops))


def tokens_to_score(total_tokens: int | None) -> float | None:
    """
    Fewer tokens is better.
    Compress large values with log1p so the penalty stays smooth.
    """
    if total_tokens is None:
        return None
    if total_tokens < 0:
        raise ValueError("total_tokens must be >= 0")
    return 1.0 / (1.0 + log1p(float(total_tokens)))


def clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# ---------------------------------
# Concrete score plugins
# ---------------------------------


class FileF1Score:
    name = "file_f1"
    default_enabled = True
    default_visible_to_agent = True

    def compute(self, ctx: ScoreContext) -> ScoreResult:
        return ScoreResult(
            name=self.name,
            value=clamp_01(ctx.metrics.f1),
            goal=Goal.MAXIMIZE,
            visible_to_agent=self.default_visible_to_agent,
            available=True,
            metadata={"source": "legacy_file_metrics"},
        )


class GoldHopScore:
    name = "gold_hop"
    default_enabled = True
    default_visible_to_agent = True

    def compute(self, ctx: ScoreContext) -> ScoreResult:
        value = hop_to_score(ctx.gold_min_hops)
        return ScoreResult(
            name=self.name,
            value=value,
            goal=Goal.MAXIMIZE,
            visible_to_agent=self.default_visible_to_agent,
            available=value is not None,
            metadata={"gold_min_hops": ctx.gold_min_hops},
        )


class IssueHopScore:
    name = "issue_hop"
    default_enabled = True
    default_visible_to_agent = True

    def compute(self, ctx: ScoreContext) -> ScoreResult:
        value = hop_to_score(ctx.issue_min_hops)
        return ScoreResult(
            name=self.name,
            value=value,
            goal=Goal.MAXIMIZE,
            visible_to_agent=self.default_visible_to_agent,
            available=value is not None,
            metadata={"issue_min_hops": ctx.issue_min_hops},
        )


class TokenEfficiencyScore:
    name = "token_efficiency"
    default_enabled = True
    default_visible_to_agent = True

    def compute(self, ctx: ScoreContext) -> ScoreResult:
        value = tokens_to_score(ctx.previous_total_tokens)
        return ScoreResult(
            name=self.name,
            value=value,
            goal=Goal.MAXIMIZE,
            visible_to_agent=self.default_visible_to_agent,
            available=value is not None,
            metadata={"previous_total_tokens": ctx.previous_total_tokens},
        )


SCORE_REGISTRY: dict[str, ScoreComponent] = {
    "file_f1": FileF1Score(),
    "gold_hop": GoldHopScore(),
    "issue_hop": IssueHopScore(),
    "token_efficiency": TokenEfficiencyScore(),
}


# ---------------------------------
# Composition config
# ---------------------------------


class ComposeMode(str, Enum):
    WEIGHTED_SUM = "weighted_sum"
    GEOMETRIC_MEAN = "geometric_mean"


@dataclass(frozen=True)
class ScoreConfig:
    """
    enabled:
        which score plugins should run at all

    visible_to_agent:
        which enabled scores are shown in prompts / feedback

    compose_weights:
        which scores participate in the machine scalar, and with what weights

    compose_mode:
        how to combine those selected scores
    """

    enabled: tuple[str, ...]
    visible_to_agent: tuple[str, ...]
    compose_weights: dict[str, float]
    compose_mode: ComposeMode = ComposeMode.GEOMETRIC_MEAN
    epsilon: float = 1e-6

    @staticmethod
    def default() -> "ScoreConfig":
        return ScoreConfig(
            enabled=("file_f1", "gold_hop", "issue_hop", "token_efficiency"),
            visible_to_agent=("file_f1", "gold_hop", "issue_hop", "token_efficiency"),
            compose_weights={
                "file_f1": 0.45,
                "gold_hop": 0.35,
                "issue_hop": 0.10,
                "token_efficiency": 0.10,
            },
            compose_mode=ComposeMode.GEOMETRIC_MEAN,
        )


# ---------------------------------
# Final output bundle
# ---------------------------------


@dataclass(frozen=True)
class ScoreBundle:
    """
    scores:
        every computed score result

    composed_score:
        one scalar for machine optimization / winner selection

    compose_mode:
        how the composed_score was built
    """

    scores: dict[str, ScoreResult]
    composed_score: float | None
    compose_mode: ComposeMode
    compose_weights: dict[str, float]


# ---------------------------------
# Engine
# ---------------------------------


class ScoreEngine:
    def __init__(
        self,
        registry: dict[str, ScoreComponent],
        config: ScoreConfig,
    ) -> None:
        self._registry = registry
        self._config = config

        missing_enabled = [name for name in config.enabled if name not in registry]
        if missing_enabled:
            raise KeyError(f"Unknown enabled scores: {missing_enabled}")

        missing_visible = [
            name for name in config.visible_to_agent if name not in config.enabled
        ]
        if missing_visible:
            raise ValueError(f"Visible scores must also be enabled: {missing_visible}")

        missing_composed = [
            name for name in config.compose_weights if name not in config.enabled
        ]
        if missing_composed:
            raise ValueError(
                f"Composed scores must also be enabled: {missing_composed}"
            )

        if not config.compose_weights:
            raise ValueError("compose_weights must not be empty")

    def evaluate(self, ctx: ScoreContext) -> ScoreBundle:
        results: dict[str, ScoreResult] = {}

        for name in self._config.enabled:
            component = self._registry[name]
            raw = component.compute(ctx)

            # Override visibility from config
            visible = name in self._config.visible_to_agent
            results[name] = ScoreResult(
                name=raw.name,
                value=raw.value,
                goal=raw.goal,
                visible_to_agent=visible,
                available=raw.available,
                metadata=raw.metadata,
            )

        composed_score = self._compose(results)
        return ScoreBundle(
            scores=results,
            composed_score=composed_score,
            compose_mode=self._config.compose_mode,
            compose_weights=dict(self._config.compose_weights),
        )

    def _compose(self, results: dict[str, ScoreResult]) -> float | None:
        selected: list[tuple[float, float]] = []

        for name, weight in self._config.compose_weights.items():
            result = results[name]

            # Strict semantics:
            # if a score is part of the machine scalar, it must be available
            if not result.available or result.value is None:
                return None

            value = clamp_01(result.value)
            selected.append((value, float(weight)))

        total_weight = sum(weight for _, weight in selected)
        if total_weight <= 0.0:
            raise ValueError("compose_weights must sum to > 0")

        # Normalize weights
        normalized = [(value, weight / total_weight) for value, weight in selected]

        if self._config.compose_mode == ComposeMode.WEIGHTED_SUM:
            return sum(value * weight for value, weight in normalized)

        if self._config.compose_mode == ComposeMode.GEOMETRIC_MEAN:
            product = 1.0
            eps = self._config.epsilon
            for value, weight in normalized:
                safe = max(eps, value)
                product *= safe**weight
            return product

        raise AssertionError(f"Unhandled compose mode: {self._config.compose_mode}")


# ---------------------------------
# Example usage
# ---------------------------------


def main() -> None:
    # Legacy file-level benchmark metrics
    metrics = LocalizationMetrics(
        precision=0.50,
        recall=0.25,
        f1=0.3333,
        hit=1.0,
        score=0.3333,
    )

    # Post-hoc graph + token signals
    ctx = ScoreContext(
        metrics=metrics,
        gold_min_hops=1,  # -> 0.5
        issue_min_hops=2,  # -> 0.333...
        previous_total_tokens=8000,
    )

    # You can fiddle with these from constants or CLI later.
    config = ScoreConfig(
        enabled=("file_f1", "gold_hop", "issue_hop", "token_efficiency"),
        visible_to_agent=("file_f1", "gold_hop", "issue_hop", "token_efficiency"),
        compose_weights={
            "file_f1": 0.45,
            "gold_hop": 0.35,
            "issue_hop": 0.10,
            "token_efficiency": 0.10,
        },
        compose_mode=ComposeMode.GEOMETRIC_MEAN,
    )

    engine = ScoreEngine(SCORE_REGISTRY, config)
    bundle = engine.evaluate(ctx)

    print("COMPOSED MACHINE SCORE")
    print(f"  mode:   {bundle.compose_mode.value}")
    print(f"  score:  {bundle.composed_score}")
    print(f"  weights:{bundle.compose_weights}")

    print("\nVISIBLE SCORES FOR AGENT")
    for name, result in bundle.scores.items():
        if result.visible_to_agent:
            print(
                f"  {name:16} "
                f"value={result.value!r:<12} "
                f"available={result.available:<5} "
                f"metadata={result.metadata}"
            )

    print("\nALL INTERNAL SCORES")
    for name, result in bundle.scores.items():
        print(
            f"  {name:16} "
            f"value={result.value!r:<12} "
            f"visible={result.visible_to_agent:<5} "
            f"available={result.available:<5}"
        )


if __name__ == "__main__":
    main()
