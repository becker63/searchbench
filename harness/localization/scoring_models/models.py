from __future__ import annotations

"""
Canonical localization scoring models.

This module defines the typed boundary for the localization scoring system.
It is intentionally broader than the static-graph implementation details:
the types here are meant to be shared by scoring, runtime evaluation,
reducers, telemetry, hosted baselines, and orchestration.

Design intent
-------------
The scoring pipeline is split into a few conceptual stages:

1. A caller constructs a `ScoreContext`.
   This is the raw, structured input to scoring. It contains the normalized
   predicted files, gold files, hop-distance summaries, token-usage summaries,
   and any diagnostic metadata that concrete scorers may need.

2. One or more score components compute `ScoreResult` values from that context.
   Each component is a small, deterministic unit of scoring logic such as
   "gold hop distance", "issue hop distance", or "token efficiency".

3. A scoring engine combines component results into a `ScoreBundle`.
   The bundle contains all individual score results, the composed machine score,
   the composition settings used, and high-level availability metadata.

4. Runtime and downstream systems summarize bundles into item-level and
   batch-level views.
   `ScoreBundle` is the single-item score representation used by reducers,
   telemetry, and reporting. Localization code uses the `TaskScoreSummary`
   alias for readability where a task id has been attached to the bundle.
   `BatchSummary[TItemBundle]` is the generic aggregate view over item bundles
   used for dataset-wide evaluation, orchestration control, and hosted
   summaries. Localization code uses `BatchScoreSummary`.

A few important invariants:
---------------------------
- These models are strict. Unknown top-level fields are rejected.
- The canonical control scalar is the composed score on `ScoreBundle`,
  and its batch aggregate on `BatchSummary` / `BatchScoreSummary`.
- A score can be unavailable without necessarily making the whole bundle unusable.
  Availability is represented explicitly rather than smuggled through sentinel values.
- These models should not depend on the static-graph implementation itself.
  Static-graph code may *produce* these models, but the models define a more
  general API boundary than any one scorer implementation.
"""

from enum import Enum
from typing import Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Goal(str, Enum):
    """
    Optimization direction for a single score.

    A score is not meaningful unless downstream consumers know whether
    "larger is better" or "smaller is better". This enum makes that explicit.

    Members
    -------
    MAXIMIZE
        Higher values are better. Typical examples include quality, relevance,
        or normalized goodness scores.

    MINIMIZE
        Lower values are better. Typical examples include distances, error
        terms, or costs when they are represented directly rather than inverted
        into a reward-like score.
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ComposeMode(str, Enum):
    """
    Strategy used to combine multiple component scores into one composed score.

    The scoring engine may support multiple composition modes so that the
    machine/control scalar can reflect different optimization semantics.

    Members
    -------
    WEIGHTED_SUM
        Computes a weighted linear combination of component scores.
        This is typically the easiest mode to reason about and is a good
        default when score magnitudes are already normalized.

    GEOMETRIC_MEAN
        Computes a multiplicative-style aggregate that penalizes weak
        components more strongly than a weighted sum. This can be useful
        when a single poor component should significantly reduce the final
        score, but it requires positive inputs and careful normalization.
    """

    WEIGHTED_SUM = "weighted_sum"
    GEOMETRIC_MEAN = "geometric_mean"


class ScoreContext(BaseModel):
    """
    Raw structured input passed to score components.

    `ScoreContext` is intentionally the "facts" layer of scoring rather than the
    "judgment" layer. It contains normalized task inputs and derived static-graph
    facts, but it does not itself decide what the final score should be.

    This separation is important because different components may interpret the
    same context differently. For example:
    - one scorer may reward small prediction-to-gold hop distance,
    - another may reward closeness to issue anchors,
    - another may reward lower token usage.

    Fields
    ------
    predicted_files
        The normalized predicted repo-relative files emitted by the localizer.

    gold_files
        The normalized gold repo-relative files for the task.

    gold_min_hops
        The minimum static-graph hop distance observed between predicted files
        and gold files, if such a distance could be computed.

    issue_min_hops
        The minimum static-graph hop distance observed between predicted files
        and issue-anchor nodes, if such a distance could be computed.

    per_prediction_hops
        Per-prediction hop diagnostics keyed by predicted file path.
        A value of `None` means the hop distance was not computable for that
        prediction rather than "infinite" or "bad".

    previous_total_token_count
        A derived scalar representing total token usage associated with the
        prior run or attempt. This is intended as the canonical token-usage
        input for token-based score components.

    resolved_prediction_nodes
        Mapping from predicted file path to the graph node ids that the scorer
        resolved for that prediction.

    resolved_gold_nodes
        Mapping from gold file path to the graph node ids that the scorer
        resolved for that gold file.

    issue_anchor_nodes
        The anchor node ids derived from issue title/body or other task context.

    unresolved_predictions
        Predicted files that could not be resolved into graph nodes.

    unresolved_gold_files
        Gold files that could not be resolved into graph nodes.

    repo_path
        Optional repository root path used while building the context.
        This is mostly diagnostic and should not typically affect score logic.

    diagnostics
        Arbitrary structured diagnostic payload. This is intentionally generic
        so the context can carry debugging or trace information without forcing
        every scorer to know about it.
    """

    model_config = ConfigDict(extra="forbid")

    predicted_files: tuple[str, ...] = Field(default_factory=tuple)
    gold_files: tuple[str, ...] = Field(default_factory=tuple)
    gold_min_hops: int | None = None
    issue_min_hops: int | None = None
    per_prediction_hops: Mapping[str, int | None] = Field(default_factory=dict)
    previous_total_token_count: float | None = None
    resolved_prediction_nodes: Mapping[str, tuple[str, ...]] = Field(
        default_factory=dict
    )
    resolved_gold_nodes: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)
    issue_anchor_nodes: tuple[str, ...] = Field(default_factory=tuple)
    unresolved_predictions: tuple[str, ...] = Field(default_factory=tuple)
    unresolved_gold_files: tuple[str, ...] = Field(default_factory=tuple)
    repo_path: str | None = None
    diagnostics: Mapping[str, object] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    """
    Result produced by a single score component.

    This model is the atomic unit of scoring output. A `ScoreResult` answers:
    - what score was computed,
    - whether it was actually available,
    - whether it should be shown to the agent,
    - and any explanation or metadata attached to it.

    Not every score must be available for every task. For example, token-based
    scoring may be unavailable if no token record exists, or a hop-based score
    may be unavailable if graph resolution fails. Instead of overloading `0.0`
    or some arbitrary sentinel, availability is represented explicitly.

    Fields
    ------
    name
        Stable component name, such as `"gold_hop"` or `"token_efficiency"`.

    value
        Numeric score value if the component produced one. `None` indicates the
        component did not produce a usable numeric value.

    goal
        Whether downstream systems should maximize or minimize this score.

    available
        Whether the score was meaningfully computed for this task.

    visible_to_agent
        Whether this score should be surfaced to the policy-writing / agent
        feedback layer. Hidden scores may still be used for machine scoring,
        telemetry, or diagnostics.

    reason
        Optional human-readable explanation for unavailability or other notable
        scorer behavior.

    metadata
        Arbitrary structured diagnostics attached by the component.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | None
    goal: Goal
    available: bool
    visible_to_agent: bool
    reason: str | None = None
    metadata: Mapping[str, object] = Field(default_factory=dict)


class ScoreConfig(BaseModel):
    """
    Configuration for score computation and composition.

    `ScoreConfig` answers three main questions:

    1. Which score components are enabled?
    2. Which enabled components are visible to the agent?
    3. How are enabled components combined into a composed machine score?

    This config is deliberately strict because loose scoring configuration can
    create subtle evaluation bugs. A component should never be visible unless
    it is enabled, and a component should never participate in composition
    unless it is enabled.

    Fields
    ------
    enabled
        Ordered component names that should be executed.

    visible_to_agent
        Subset of enabled component names that should be surfaced to the agent.

    compose_weights
        Mapping from component name to composition weight. Only components named
        here participate in the composed score.

    compose_mode
        The mathematical strategy used to combine composed component scores.

    epsilon
        Small positive constant used to stabilize computations in composition
        modes that need it, such as geometric mean.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: tuple[str, ...]
    visible_to_agent: tuple[str, ...]
    compose_weights: Mapping[str, float]
    compose_mode: ComposeMode = ComposeMode.WEIGHTED_SUM
    epsilon: float = 1e-6

    @model_validator(mode="after")
    def _validate_config(self) -> "ScoreConfig":
        """
        Enforce internal consistency of the scoring configuration.

        Validation rules:
        - at least one component must be enabled,
        - at least one component must participate in composition,
        - visible component names must be a subset of enabled component names,
        - composed component names must be a subset of enabled component names,
        - total composition weight must be positive,
        - epsilon must be strictly positive.

        These checks make configuration errors fail fast at construction time
        rather than producing confusing runtime behavior later.
        """
        enabled = set(self.enabled)
        visible = set(self.visible_to_agent)
        composed = set(self.compose_weights)
        if not self.enabled:
            raise ValueError("enabled must not be empty")
        if not self.compose_weights:
            raise ValueError("compose_weights must not be empty")
        if not visible <= enabled:
            raise ValueError(
                f"Visible scores must also be enabled: {sorted(visible - enabled)}"
            )
        if not composed <= enabled:
            raise ValueError(
                f"Composed scores must also be enabled: {sorted(composed - enabled)}"
            )
        total_weight = sum(float(weight) for weight in self.compose_weights.values())
        if total_weight <= 0.0:
            raise ValueError("compose_weights must sum to > 0")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        return self

    @classmethod
    def default(cls) -> "ScoreConfig":
        """
        Return the default scoring configuration.

        Current default behavior:
        - all three canonical components are enabled,
        - all three are visible to the agent,
        - only `gold_hop` and `issue_hop` participate in composition,
        - composition uses weighted sum,
        - `gold_hop` has the dominant weight.

        This default expresses an important policy choice:
        token efficiency is tracked and surfaced, but does not control the main
        machine score unless a caller explicitly opts into that behavior.
        """
        return cls(
            enabled=("gold_hop", "issue_hop", "token_efficiency"),
            visible_to_agent=("gold_hop", "issue_hop", "token_efficiency"),
            compose_weights={
                "gold_hop": 0.75,
                "issue_hop": 0.25,
            },
            compose_mode=ComposeMode.WEIGHTED_SUM,
        )


class ScoreComponent(Protocol):
    """
    Structural protocol implemented by concrete score components.

    Concrete scorers do not need to inherit from a base class as long as they
    satisfy this protocol. This keeps component implementations lightweight
    while still giving the engine and type checker a consistent interface.

    Required attributes
    -------------------
    name
        Stable identifier for the component.

    default_enabled
        Whether this component should normally be enabled in default configs.

    default_visible_to_agent
        Whether this component should normally be shown to the agent.

    Required method
    ---------------
    compute(ctx: ScoreContext) -> ScoreResult
        Compute the component result deterministically from the given context.
    """

    name: str
    default_enabled: bool
    default_visible_to_agent: bool

    def compute(self, ctx: ScoreContext) -> ScoreResult:
        """
        Compute this component's score from a `ScoreContext`.

        Implementations should:
        - be deterministic,
        - return a `ScoreResult` with a stable `name`,
        - mark uncomputable situations via `available=False`,
        - avoid raising for ordinary "not enough data" cases when a structured
          unavailable result is sufficient.
        """
        ...


__all__ = [
    "ComposeMode",
    "Goal",
    "ScoreComponent",
    "ScoreConfig",
    "ScoreContext",
    "ScoreResult",
]
