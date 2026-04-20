from __future__ import annotations

from pathlib import Path

import pytest

from harness.utils import type_loader
from harness.pipeline.types import StepResult
from harness.prompts import (
    SystemPromptContext,
    WriterFeedbackFinding,
    WriterPromptContext,
    WriterScoreComponent,
    WriterScoreSummary,
    build_ic_system_prompt,
    build_jc_system_prompt,
    build_writer_optimization_brief,
    build_writer_prompt,
)
from harness.prompts import template_loader
from harness.utils.openai_schema import ChatCompletionToolParam


def test_writer_prompt_renders_jinja(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    template_path = prompts_dir / "policy_writer.jinja"
    (prompts_dir / "policy_writer_system.jinja").write_text("system", encoding="utf-8")
    template_path.write_text(
        "Policy: {{ current_policy }} | Tests: {{ tests }} | Findings: {{ optimization_brief.findings[0].message }}",
        encoding="utf-8",
    )
    monkeypatch.setattr(template_loader, "find_repo_root", lambda: tmp_path)
    dummy_ctx = type_loader.FrontierContext(
        signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
        graph_models="",
        types="",
        examples="",
        notes="",
    )

    rendered = build_writer_prompt(
        current_policy="code_v1",
        failure_context="ctx",
        tests="tests-content",
        frontier_context="ctx",
        frontier_context_details=dummy_ctx,
        optimization_brief=build_writer_optimization_brief(
            findings=[WriterFeedbackFinding(kind="feedback", message="fb")],
            comparison_summary="N/A",
            diff_summary="diff",
            diff_guidance="hint2",
            guidance=["hint"],
        ),
    )
    assert rendered.system == "system"
    assert "code_v1" in rendered.user
    assert "tests-content" in rendered.user
    assert "fb" in rendered.user


def test_real_writer_prompt_includes_selection_signature():
    rendered = build_writer_prompt(
        current_policy="def frontier_priority(node, graph, step):\n    return 0",
        failure_context="",
        tests="",
        frontier_context="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
        frontier_context_details=type_loader.FrontierContext(
            signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
            graph_models="class GraphNode:\n    pass\nclass Graph:\n    ...",
            types="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
            examples="from iterative_context.graph_models import Graph, GraphNode\n"
            "def frontier_priority(node: GraphNode, graph: Graph, step: int) -> float:\n    return 1.0\n",
            notes="Traversal calls frontier_priority(node, graph, step).",
        ),
        optimization_brief=build_writer_optimization_brief(comparison_summary="N/A"),
    )
    assert "frontier priority function" in rendered.system
    assert "SelectionCallable" in rendered.user
    assert "GraphNode" in rendered.user
    assert "graph" in rendered.user


def test_runner_prompt_renders_jinja(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "ic_system.jinja").write_text("Tools: {{ available_tools }}", encoding="utf-8")
    (prompts_dir / "jc_system.jinja").write_text("JC Tools: {{ available_tools }}", encoding="utf-8")
    monkeypatch.setattr(template_loader, "find_repo_root", lambda: tmp_path)

    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "inspect_file",
                "description": "Inspect a file",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    ic_prompt = build_ic_system_prompt(tools)
    jc_prompt = build_jc_system_prompt(tools)
    assert "inspect_file" in ic_prompt
    assert "inspect_file" in jc_prompt


def test_real_ic_prompt_teaches_resolve_then_expand_workflow():
    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "resolve",
                "description": "Resolve an anchor",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "expand",
                "description": "Expand graph neighborhood",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    prompt = build_ic_system_prompt(tools)

    assert "`resolve` to find" in prompt
    assert "`expand` as the primary exploration primitive" in prompt
    assert "call `expand` again" in prompt
    assert "not just the first `resolve` lookup" in prompt


def test_writer_prompt_renders_score_summary_as_optimization_contract():
    rendered = build_writer_prompt(
        current_policy="def frontier_priority(node, graph, step):\n    return 0",
        failure_context="",
        tests="",
        frontier_context="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
        frontier_context_details=type_loader.FrontierContext(
            signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
            graph_models="class GraphNode:\n    pass\nclass Graph:\n    ...",
            types="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
            examples="",
            notes="",
        ),
        optimization_brief=build_writer_optimization_brief(
            score_summary=WriterScoreSummary(
                primary_name="composed_score",
                primary_score=0.42,
                composed_score=0.42,
                optimization_goal="maximize",
                components=[
                    WriterScoreComponent(
                        name="gold_hop",
                        value=0.5,
                        goal="maximize",
                        weight=0.75,
                        rationale="closer to changed files is better",
                        guard="floor: 0.2",
                    )
                ],
            ),
            comparison_summary="N/A",
        ),
    )
    assert "## Scoring objective" in rendered.user
    assert "Primary scalar: composed_score" in rendered.user
    assert "Current composed score: 0.42" in rendered.user
    assert "gold_hop" in rendered.user
    assert "weight=0.75" in rendered.user
    assert "Improve the composed objective" in rendered.user


def test_writer_prompt_validation_rejects_extra_fields():
    with pytest.raises(Exception):
        WriterPromptContext.model_validate(
            {
                "current_policy": "code",
                "failure_context": "",
                "tests": "",
                "frontier_context": "",
                "frontier_context_details": type_loader.FrontierContext(
                    signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
                    graph_models="",
                    types="",
                    examples="",
                    notes="",
                ),
                "optimization_brief": build_writer_optimization_brief(),
                "extra_field": "nope",
            }
        )


def test_writer_prompt_does_not_use_user_delimiter():
    rendered = build_writer_prompt(
        current_policy="def frontier_priority(node, graph, step):\n    return 0",
        failure_context="",
        tests="",
        frontier_context="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
        frontier_context_details=type_loader.FrontierContext(
            signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
            graph_models="",
            types="",
            examples="",
            notes="",
        ),
        optimization_brief=build_writer_optimization_brief(),
    )
    assert "===USER===" not in rendered.system
    assert "===USER===" not in rendered.user


def test_step_result_rejects_extra_fields():
    with pytest.raises(Exception):
        StepResult.model_validate(
            {
                "name": "x",
                "success": True,
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "unexpected": "nope",
            }
        )


def test_system_prompt_context_requires_available_tools():
    with pytest.raises(Exception):
        SystemPromptContext.model_validate({"available_tools": ""})
