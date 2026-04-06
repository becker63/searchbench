"""
Loop package: optimizer coordination, state machines, listeners, and models.
Exports the public loop API for callers outside the package. Attribute access
is forwarded to ``harness.loop.loop`` so monkeypatching this package also
updates the implementation module.
"""

# pyright: reportPrivateUsage=false

from importlib import import_module

_impl = import_module("harness.loop.loop")

from .loop import (  # noqa: E402
    _clean_policy_code,
    _hash_tests_dir,
    _read_policy,
    _write_policy,
    emit_score,
    evaluate_policy_on_item,
    index_folder,
    load_policy,
    build_scorer_context,
    load_graph_models,
    prepare_iteration_tasks,
    run_loop,
    run_policy_pipeline,
)
from .loop_machine import OptimizationStateMachine, RepairStateMachine
from .loop_types import (
    AcceptedPolicy,
    AcceptedPolicyMeta,
    EvaluationResult,
    FailedRepairDetails,
    FeedbackPackage,
    IterationRecord,
    IterationTasks,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
    RepairOutcome,
    RunMetadata,
    TaskPayload,
    iteration_record_to_public_dict,
)
from .runner_agent import run_ic_iteration

__all__ = [
    "run_loop",
    "prepare_iteration_tasks",
    "evaluate_policy_on_item",
    "_read_policy",
    "_write_policy",
    "_clean_policy_code",
    "index_folder",
    "emit_score",
    "load_policy",
    "load_graph_models",
    "build_scorer_context",
    "run_policy_pipeline",
    "run_ic_iteration",
    "_hash_tests_dir",
    "OptimizationStateMachine",
    "RepairStateMachine",
    "AcceptedPolicy",
    "AcceptedPolicyMeta",
    "EvaluationResult",
    "FailedRepairDetails",
    "FeedbackPackage",
    "IterationTasks",
    "IterationRecord",
    "LoopContext",
    "LoopDependencies",
    "OptimizationMachineModel",
    "PreparedTasks",
    "RepairContext",
    "RepairMachineModel",
    "RepairOutcome",
    "RunMetadata",
    "TaskPayload",
    "iteration_record_to_public_dict",
]


def __getattr__(name: str) -> object:
    return getattr(_impl, name)


def __setattr__(name: str, value: object) -> None:
    if name.startswith("__"):
        globals()[name] = value
        return
    setattr(_impl, name, value)
    globals()[name] = value
