from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from harness.utils.openai_schema import ChatCompletionToolParam

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

if TYPE_CHECKING:
    from harness.scoring import ScoreBundle
    from harness.utils.type_loader import FrontierContext

_WRITER_SYSTEM_TEMPLATE = "policy_writer_system.jinja"
_WRITER_USER_TEMPLATE = "policy_writer.jinja"


def _format_available_tools(tool_specs: list[ChatCompletionToolParam]) -> str:
    names: list[str] = []
    for spec in tool_specs:
        func = spec.get("function")
        name = func.get("name") if isinstance(func, Mapping) else None
        if isinstance(name, str) and name:
            names.append(name)
    if not names:
        return "- none"
    return "\n".join(f"- {name}" for name in names)


def build_ic_system_prompt(tool_specs: list[ChatCompletionToolParam]) -> str:
    context = SystemPromptContext(available_tools=_format_available_tools(tool_specs))
    return render_prompt_template("ic_system.jinja", context.model_dump())


def build_jc_system_prompt(tool_specs: list[ChatCompletionToolParam]) -> str:
    context = SystemPromptContext(available_tools=_format_available_tools(tool_specs))
    return render_prompt_template("jc_system.jinja", context.model_dump())


def build_writer_prompt(
    *,
    current_policy: str,
    failure_context: str,
    tests: str,
    frontier_context: str,
    frontier_context_details: "FrontierContext",
    optimization_brief: WriterOptimizationBrief,
) -> RenderedPrompt:
    context = WriterPromptContext.model_validate(
        {
            "current_policy": current_policy,
            "failure_context": failure_context,
            "tests": tests,
            "frontier_context": frontier_context,
            "frontier_context_details": frontier_context_details,
            "optimization_brief": optimization_brief,
        }
    )
    payload = context.model_dump(mode="json")
    return RenderedPrompt(
        system=render_prompt_template(_WRITER_SYSTEM_TEMPLATE, payload).strip(),
        user=render_prompt_template(_WRITER_USER_TEMPLATE, payload).strip(),
    )


def build_writer_optimization_brief(
    *,
    score_summary: WriterScoreSummary | None = None,
    findings: Sequence[WriterFeedbackFinding] = (),
    comparison_summary: str | None = None,
    diff_summary: str | None = None,
    diff_guidance: str | None = None,
    guidance: Sequence[str] = (),
    mode: str = "repair_then_optimize",
) -> WriterOptimizationBrief:
    return WriterOptimizationBrief(
        score_summary=score_summary,
        findings=list(findings),
        comparison=WriterComparisonSummary(summary=comparison_summary),
        diff=WriterDiffSummary(summary=diff_summary, guidance=diff_guidance),
        guidance=WriterGuidance(mode=mode, instructions=list(guidance)),
    )


def build_writer_score_summary_from_bundle(bundle: "ScoreBundle") -> WriterScoreSummary:
    """Translate score internals into a concise writer-facing contract."""

    components: list[WriterScoreComponent] = []
    for name, result in bundle.results.items():
        if not result.visible_to_agent:
            continue
        metadata = result.metadata or {}
        guard = _metadata_text(metadata, ("guard", "floor", "minimum", "maximum"))
        components.append(
            WriterScoreComponent(
                name=name,
                value=result.value,
                goal=getattr(result.goal, "value", str(result.goal)),
                weight=float(bundle.compose_weights[name]) if name in bundle.compose_weights else None,
                rationale=result.reason,
                guard=guard,
            )
        )
    return WriterScoreSummary(
        primary_name="composed_score",
        primary_score=bundle.composed_score,
        composed_score=bundle.composed_score,
        optimization_goal="maximize",
        components=components,
    )


def _metadata_text(metadata: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    parts: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            parts.append(f"{key}: {value}")
    return "; ".join(parts) if parts else None


__all__ = [
    "build_ic_system_prompt",
    "build_jc_system_prompt",
    "build_writer_optimization_brief",
    "build_writer_prompt",
    "build_writer_score_summary_from_bundle",
]
