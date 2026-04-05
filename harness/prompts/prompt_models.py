from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class WriterPromptContext(BaseModel):
    """Strict context for the policy writer prompt."""

    model_config = ConfigDict(extra="forbid")

    current_policy: str
    failure_context: str
    feedback_text: str
    comparison_summary: str
    guidance_hint: str
    diff_str: str
    diff_hint: str
    tests: str
    scoring_context: str


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
