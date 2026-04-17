"""Public prompt construction API."""

from .builders import (
    build_ic_system_prompt,
    build_jc_system_prompt,
    build_writer_optimization_brief,
    build_writer_prompt,
    build_writer_score_summary_from_bundle,
)
from .prompt_models import (
    RenderedPrompt,
    SystemPromptContext,
    WriterComparisonSummary,
    WriterDiffSummary,
    WriterFeedbackFinding,
    WriterGuidance,
    WriterOptimizationBrief,
    WriterPromptContext,
    WriterScoreComponent,
    WriterScoreSummary,
)
from .template_loader import render_prompt_template

__all__ = [
    "RenderedPrompt",
    "SystemPromptContext",
    "WriterComparisonSummary",
    "WriterDiffSummary",
    "WriterFeedbackFinding",
    "WriterGuidance",
    "WriterOptimizationBrief",
    "WriterPromptContext",
    "WriterScoreComponent",
    "WriterScoreSummary",
    "build_ic_system_prompt",
    "build_jc_system_prompt",
    "build_writer_optimization_brief",
    "build_writer_prompt",
    "build_writer_score_summary_from_bundle",
    "render_prompt_template",
]
