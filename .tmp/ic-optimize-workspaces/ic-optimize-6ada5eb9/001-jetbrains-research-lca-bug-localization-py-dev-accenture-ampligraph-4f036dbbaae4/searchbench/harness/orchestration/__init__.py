"""Orchestration: explicit optimize loops, repair state, and typed contracts."""
# pyright: reportImportCycles=false

from .machine import RepairStateMachine
from .optimize_loop import DEFAULT_OPTIMIZATION_ITERATIONS, OptimizeTaskLoop, run_loop
from .types import (
    AcceptedPolicy,
    AcceptedPolicyMeta,
    EvaluationResult,
    FailedRepairDetails,
    FeedbackPackage,
    IterationRecord,
    IterationTasks,
    LoopContext,
    LoopDependencies,
    OptimizationIterationState,
    EvaluationPhaseResult,
    ReducerSummary,
    WriterInputContext,
    IterationOutcome,
    PreparedTasks,
    RepairContext,
    RepairMachineModel,
    RepairOutcome,
    RunMetadata,
    iteration_record_to_public_dict,
)

__all__ = [
    "run_loop",
    "DEFAULT_OPTIMIZATION_ITERATIONS",
    "OptimizeTaskLoop",
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
    "OptimizationIterationState",
    "EvaluationPhaseResult",
    "ReducerSummary",
    "WriterInputContext",
    "IterationOutcome",
    "PreparedTasks",
    "RepairContext",
    "RepairMachineModel",
    "RepairOutcome",
    "RunMetadata",
    "iteration_record_to_public_dict",
]
