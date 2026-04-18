from __future__ import annotations

import inspect
from typing import get_type_hints

from harness.agents.common import serialize_lca_task_for_prompt
from harness.agents.localizer import run_ic_iteration, run_jc_iteration
from harness.localization.models import (
    LCAContext,
    LCAGold,
    LCATask,
    LCATaskIdentity,
    normalize_lca_task,
)
from harness import orchestration
from harness.orchestration import dependencies as orchestration_dependencies
from harness.orchestration import runtime
from harness.orchestration import types as orchestration_types

DELETED_TASK_WRAPPERS = (
    "TaskPayload",
    "ICTaskPayload",
    "JCTaskPayload",
    "AgentTaskPayload",
    "AgentTask",
)


def _task() -> LCATask:
    return LCATask(
        identity=LCATaskIdentity(
            dataset_name="lca",
            dataset_config="py",
            dataset_split="dev",
            repo_owner="o",
            repo_name="r",
            base_sha="abc",
        ),
        context=LCAContext(issue_title="bug", issue_body="details"),
        gold=LCAGold(changed_files=["a.py"]),
        repo="repo",
    )


def test_duplicate_internal_task_wrappers_are_not_exported_from_orchestration() -> None:
    for name in DELETED_TASK_WRAPPERS:
        assert not hasattr(orchestration_types, name)
        assert not hasattr(orchestration, name)


def test_localization_module_exports_single_internal_task_model() -> None:
    assert LCATask.__name__ == "LCATask"
    assert normalize_lca_task({"changed_files": []}).__class__ is LCATask


def test_orchestration_records_reference_lca_task_directly() -> None:
    assert orchestration_types.RunMetadata.model_fields["task"].annotation is LCATask
    assert orchestration_types.PreparedTasks.model_fields["task"].annotation is LCATask
    assert orchestration_types.IterationTasks.model_fields["task"].annotation is LCATask
    assert orchestration_types.LoopContext.model_fields["task"].annotation is LCATask
    assert inspect.signature(orchestration_dependencies.prepare_iteration_tasks).parameters["task"].annotation in {LCATask, "LCATask"}
    assert inspect.signature(runtime.evaluate_policy_on_item).parameters["task"].annotation in {LCATask, "LCATask"}
    assert inspect.signature(runtime.run_loop).parameters["task"].annotation in {LCATask, "LCATask"}


def test_ic_and_jc_share_lca_task_input_type() -> None:
    assert inspect.signature(run_ic_iteration).parameters["task"].name == "task"
    assert inspect.signature(run_jc_iteration).parameters["task"].name == "task"
    assert get_type_hints(run_ic_iteration)["task"] is LCATask
    assert get_type_hints(run_jc_iteration)["task"] is LCATask


def test_prompt_serializer_is_one_way_edge_payload() -> None:
    task = _task()
    payload = serialize_lca_task_for_prompt(task)

    assert payload["identity"] == task.identity.model_dump()
    assert payload["context"] == task.context.model_dump()
    assert payload["repo"] == task.repo
    assert "gold" not in payload
    assert "changed_files" not in payload
