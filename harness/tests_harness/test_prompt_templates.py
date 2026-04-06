from __future__ import annotations

from pathlib import Path

import pytest

import harness.loop.runner_agent as runner
import harness.writer as writer
from harness.utils import type_loader
from harness.pipeline.types import StepResult
from harness.prompts import SystemPromptContext, WriterPromptContext
from harness.utils import template_loader
from harness.utils.openai_schema import OpenAITool


def test_writer_prompt_renders_jinja(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    template_path = prompts_dir / "policy_writer.jinja"
    template_path.write_text("Policy: {{ current_policy }} | Tests: {{ tests }} | Feedback: {{ feedback_text }}", encoding="utf-8")
    monkeypatch.setattr(template_loader, "find_repo_root", lambda: tmp_path)
    dummy_ctx = type_loader.ScorerContext(
        signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
        graph_models="",
        types="",
        examples="",
        notes="",
    )

    rendered = writer._render_writer_prompt(
        current_policy="code_v1",
        failure_context="ctx",
        feedback_str="fb",
        feedback={},
        comparison_summary="N/A",
        guidance_hint="hint",
        diff_str="diff",
        diff_hint="hint2",
        tests="tests-content",
        scoring_context="ctx",
        scoring_context_details=dummy_ctx,
    )
    assert "code_v1" in rendered
    assert "tests-content" in rendered
    assert "fb" in rendered


def test_real_writer_prompt_includes_selection_signature():
    rendered = writer._render_writer_prompt(
        current_policy="def score(node, graph, step):\n    return 0",
        failure_context="",
        feedback_str="",
        feedback={},
        comparison_summary="N/A",
        guidance_hint="",
        diff_str="",
        diff_hint="",
        tests="",
        scoring_context="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
        scoring_context_details=type_loader.ScorerContext(
            signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
            graph_models="class GraphNode:\n    pass\nclass Graph:\n    ...",
            types="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
            examples="from iterative_context.graph_models import Graph, GraphNode\n"
            "def score(node: GraphNode, graph: Graph, step: int) -> float:\n    return 1.0\n",
            notes="Traversal calls score(node, graph, step).",
        ),
    )
    assert "SelectionCallable" in rendered
    assert "GraphNode" in rendered
    assert "graph" in rendered


def test_runner_prompt_renders_jinja(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "ic_system.jinja").write_text("Tools: {{ available_tools }}", encoding="utf-8")
    (prompts_dir / "jc_system.jinja").write_text("JC Tools: {{ available_tools }}", encoding="utf-8")
    monkeypatch.setattr(template_loader, "find_repo_root", lambda: tmp_path)

    tools: list[OpenAITool] = [
        {
            "type": "function",
            "function": {
                "name": "inspect_file",
                "description": "Inspect a file",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    ic_prompt = runner._build_ic_system_prompt(tools)
    jc_prompt = runner._build_jc_system_prompt(tools)
    assert "inspect_file" in ic_prompt
    assert "inspect_file" in jc_prompt


def test_writer_prompt_validation_rejects_extra_fields():
    with pytest.raises(Exception):
        WriterPromptContext.model_validate(
            {
                "current_policy": "code",
                "failure_context": "",
                "feedback_text": "fb",
                "comparison_summary": "N/A",
                "guidance_hint": "hint",
                "diff_str": "",
                "diff_hint": "",
                "tests": "",
                "scoring_context": "",
                "scoring_context_details": type_loader.ScorerContext(
                    signature="SelectionCallable = Callable[[GraphNode, Graph, int], float]",
                    graph_models="",
                    types="",
                    examples="",
                    notes="",
                ),
                "extra_field": "nope",
            }
        )


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
