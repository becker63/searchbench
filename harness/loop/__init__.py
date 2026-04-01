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
    index_folder,
    evaluate_policy_on_item,
    prepare_iteration_tasks,
    run_loop,
    emit_score,
    emit_score_for_handle,
    load_policy,
    run_policy_pipeline,
    run_ic_iteration,
)
from .loop_machine import OptimizationStateMachine, RepairStateMachine
from .loop_types import (
    AcceptedPolicy,
    AcceptedPolicyMeta,
    EvaluationResult,
    FailedRepairDetails,
    FeedbackPackage,
    IterationRecord,
    LoopContext,
    LoopDependencies,
    OptimizationMachineModel,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
    RepairOutcome,
)
from .loop_types import iteration_record_to_public_dict

__all__ = [
    "run_loop",
    "prepare_iteration_tasks",
    "evaluate_policy_on_item",
    "_read_policy",
    "_write_policy",
    "_clean_policy_code",
    "index_folder",
    "emit_score",
    "emit_score_for_handle",
    "load_policy",
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
    "IterationRecord",
    "LoopContext",
    "LoopDependencies",
    "OptimizationMachineModel",
    "PreparedTasks",
    "RepairContext",
    "RepairMachineModel",
    "RepairOutcome",
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
