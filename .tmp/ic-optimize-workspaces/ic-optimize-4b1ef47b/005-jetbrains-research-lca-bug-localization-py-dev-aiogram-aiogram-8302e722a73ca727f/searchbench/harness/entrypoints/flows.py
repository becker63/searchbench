from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FlowNarrative(BaseModel):
    """Human-readable map of a CLI flow; not an execution engine."""

    model_config = ConfigDict(frozen=True)

    command_name: str
    purpose: str
    inputs: tuple[str, ...]
    steps: tuple[str, ...]
    outputs: tuple[str, ...]


JC_RUN_NARRATIVE = FlowNarrative(
    command_name="jc run",
    purpose="Run the JCodeMunch localization evaluator over the fixed project task set.",
    inputs=(
        "fixed project dataset",
        "selection window",
        "optional worktree refresh",
        "run session",
    ),
    steps=(
        "create session",
        "load fixed task set",
        "optionally refresh worktrees",
        "run JC localizer/evaluator flow",
        "report results",
    ),
    outputs=(
        "selection summary",
        "cost projection",
        "JC localization scores",
        "materialization summaries",
    ),
)

IC_RUN_NARRATIVE = FlowNarrative(
    command_name="ic run",
    purpose="Run the Iterative Context localization evaluator over the fixed project task set.",
    inputs=(
        "fixed project dataset",
        "selection window",
        "optional worktree refresh",
        "run session",
    ),
    steps=(
        "create session",
        "load fixed task set",
        "optionally refresh worktrees",
        "run IC localizer/evaluator flow",
        "report results",
    ),
    outputs=(
        "selection summary",
        "cost projection",
        "IC localization scores",
        "materialization summaries",
    ),
)

IC_OPTIMIZE_NARRATIVE = FlowNarrative(
    command_name="ic optimize",
    purpose="Optimize the IC policy through evaluator feedback, writer updates, isolated candidate workspaces, and pipeline checks.",
    inputs=(
        "fixed project dataset",
        "selection window",
        "optional worktree refresh",
        "optimize run id",
    ),
    steps=(
        "create optimize run id",
        "load fixed task set",
        "optionally refresh worktrees",
        "run selected optimization tasks in isolated workspaces up to the worker limit",
        "evaluate current IC behavior",
        "build writer feedback from evaluation",
        "generate candidate policy update",
        "derive writer replay session for the task",
        "run pipeline correctness checks",
        "accept or reject the update",
        "report iteration outcome",
    ),
    outputs=(
        "selection summary",
        "cost projection",
        "optimization iteration records",
        "accepted or rejected policy outcome",
    ),
)

FLOW_NARRATIVES = {
    narrative.command_name: narrative
    for narrative in (JC_RUN_NARRATIVE, IC_RUN_NARRATIVE, IC_OPTIMIZE_NARRATIVE)
}


def get_flow_narrative(command_name: str) -> FlowNarrative:
    return FLOW_NARRATIVES[command_name]


def flow_help(narrative: FlowNarrative) -> str:
    steps = " -> ".join(narrative.steps)
    return f"{narrative.purpose}\n\nFlow: {steps}"
