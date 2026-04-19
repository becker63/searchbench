from __future__ import annotations

from harness.agents.common import normalize_token_usage
from harness.telemetry.observability_models import (
    RunSummary,
    TaskRunSummary,
    TokenUsageBreakdown,
)


def _summary(backend: str, status: str = "success") -> TaskRunSummary:
    return TaskRunSummary(
        backend=backend,
        identity="task-1",
        repo="/repo",
        status=status,
        stage=None if status == "success" else "backend_init",
        message=None if status == "success" else "boom",
        step_count=2,
        generation_count=3,
        tool_calls_count=4,
        successful_tool_calls_count=3,
        failed_tool_calls_count=1,
        predicted_files_count=1 if status == "success" else 0,
        duration_seconds=5.0,
        tokens=TokenUsageBreakdown(
            fresh_input_tokens=10,
            cached_input_tokens=20,
            total_input_tokens=30,
            output_tokens=4,
            total_tokens=34,
        ),
    )


def test_normalize_token_usage_with_cached_tokens():
    usage = normalize_token_usage(
        {"input": 10, "input_cached_tokens": 20, "output": 4, "total": 34}
    )

    assert usage.fresh_input_tokens == 10
    assert usage.cached_input_tokens == 20
    assert usage.total_input_tokens == 30
    assert usage.output_tokens == 4
    assert usage.total_tokens == 34


def test_normalize_token_usage_without_cached_tokens_computes_total():
    usage = normalize_token_usage({"input": 10, "output": 4})

    assert usage.fresh_input_tokens == 10
    assert usage.cached_input_tokens == 0
    assert usage.total_input_tokens == 10
    assert usage.output_tokens == 4
    assert usage.total_tokens == 14


def test_task_summary_schema_parity_for_backends():
    ic_success = _summary("ic").model_dump()
    jc_success = _summary("jc").model_dump()
    ic_failure = _summary("ic", status="failure").model_dump()
    jc_failure = _summary("jc", status="failure").model_dump()

    assert set(ic_success) == set(jc_success)
    assert set(ic_failure) == set(jc_failure)


def test_run_summary_aggregates_from_task_summaries():
    summaries = [_summary("ic"), _summary("ic", status="failure")]

    run = RunSummary.from_task_summaries(
        backend="ic",
        selected_tasks_count=3,
        task_summaries=summaries,
    )

    assert run.selected_tasks_count == 3
    assert run.completed_tasks_count == 2
    assert run.successful_tasks_count == 1
    assert run.failed_tasks_count == 1
    assert run.success_rate == 0.5
    assert run.tokens.total_tokens == 68
    assert run.average_tokens_per_task.total_tokens == 34
    assert run.total_tool_calls_count == 8
    assert run.total_successful_tool_calls_count == 6
    assert run.total_failed_tool_calls_count == 2
    assert run.average_tool_calls_per_task == 4
    assert run.total_generations_count == 6
    assert run.average_generations_per_task == 3
    assert run.total_duration_seconds == 10
    assert run.average_task_duration_seconds == 5
    assert run.failure_stages == {"backend_init": 1}
