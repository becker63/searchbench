from __future__ import annotations

from harness.agents.writer import generate_policy
from harness.pipeline.types import PipelineClassification
from harness.prompts import WriterOptimizationBrief
from harness.utils.type_loader import FrontierContext


def clean_policy_code(code: str) -> str:
    if "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if code.startswith("python"):
                code = code[len("python") :].lstrip()
    return code.strip()


def attempt_policy_generation(
    *,
    initial_policy: str,
    tests: str,
    frontier_context: str,
    frontier_context_details: FrontierContext,
    optimization_brief: WriterOptimizationBrief,
    failure_context_str: str | None,
    repair_attempt: int,
    parent_trace: object | None = None,
    pipeline_feedback: PipelineClassification | None = None,
) -> tuple[str | None, str | None]:
    try:
        feedback = {}
        if pipeline_feedback is not None:
            feedback = {"pipeline_feedback": pipeline_feedback.model_dump()}
        new_code = generate_policy(
            current_policy=initial_policy,
            tests=tests,
            frontier_context=frontier_context,
            frontier_context_details=frontier_context_details,
            optimization_brief=optimization_brief,
            feedback=feedback,
            parent_trace=parent_trace,
            failure_context=failure_context_str,
            repair_attempt=repair_attempt,
        )
        return clean_policy_code(new_code), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
