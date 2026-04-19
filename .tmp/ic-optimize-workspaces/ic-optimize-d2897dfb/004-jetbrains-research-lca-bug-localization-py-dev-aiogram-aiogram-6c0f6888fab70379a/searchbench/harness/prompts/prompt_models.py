from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator
from harness.utils.type_loader import FrontierContext


class RenderedPrompt(BaseModel):
    """Rendered multi-message prompt content."""

    model_config = ConfigDict(extra="forbid")

    system: str
    user: str


class WriterScoreComponent(BaseModel):
    """Writer-facing description of one score component."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | None = None
    goal: str | None = None
    weight: float | None = None
    rationale: str | None = None
    guard: str | None = None


class WriterScoreSummary(BaseModel):
    """Concise optimization contract for policy writing."""

    model_config = ConfigDict(extra="forbid")

    primary_name: str | None = None
    primary_score: float | None = None
    composed_score: float | None = None
    optimization_goal: str = "maximize"
    components: list[WriterScoreComponent] = Field(default_factory=list)


class WriterFeedbackFinding(BaseModel):
    """One explicit writer-facing evaluation finding."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    message: str
    value: str | float | int | bool | None = None
    severity: str | None = None
    source: str | None = None


class WriterComparisonSummary(BaseModel):
    """Writer-facing baseline/comparison context."""

    model_config = ConfigDict(extra="forbid")

    summary: str | None = None


class WriterDiffSummary(BaseModel):
    """Writer-facing score/pipeline delta context."""

    model_config = ConfigDict(extra="forbid")

    summary: str | None = None
    guidance: str | None = None


class WriterGuidance(BaseModel):
    """Explicit next-step steering for the writer."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "repair_then_optimize"
    instructions: list[str] = Field(default_factory=list)


class WriterOptimizationBrief(BaseModel):
    """Coherent typed contract for writer prompt optimization."""

    model_config = ConfigDict(extra="forbid")

    score_summary: WriterScoreSummary | None = None
    findings: list[WriterFeedbackFinding] = Field(default_factory=list)
    comparison: WriterComparisonSummary = Field(default_factory=WriterComparisonSummary)
    diff: WriterDiffSummary = Field(default_factory=WriterDiffSummary)
    guidance: WriterGuidance = Field(default_factory=WriterGuidance)


class WriterPromptContext(BaseModel):
    """Strict context for the policy writer prompt."""

    model_config = ConfigDict(extra="forbid")

    current_policy: str
    failure_context: str
    tests: str
    frontier_context: str
    frontier_context_details: FrontierContext
    optimization_brief: WriterOptimizationBrief


class SystemPromptContext(BaseModel):
    """Strict context for runner system prompts."""

    model_config = ConfigDict(extra="forbid")

    available_tools: str

    @field_validator("available_tools")
    @classmethod
    def _ensure_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("available_tools must be non-empty")
        return value


class BackendPromptContext(BaseModel):
    """Explicit context for backend-specific prompts."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    available_tools: str
