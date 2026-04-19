## Why

The CLI still exposes an older benchmark-era model: `baseline`, `experiment`, dataset-oriented inputs, and baseline cache invalidation. The project’s real operator-facing model is now flow-oriented: run JC, run IC, and optimize IC, with the IC optimization loop evaluating behavior, turning evaluation into writer-facing feedback, generating a policy candidate, and validating it through pipeline checks.

## What Changes

- **BREAKING**: Redesign the public CLI surface around project flows instead of historical dataset/baseline infrastructure.
- Replace the main public commands:
  - from `baseline`
  - from `experiment`
  - toward `jc run`
  - toward `ic run`
  - toward `ic optimize`
- Keep `sync-hf` available for now as ingestion tooling, but treat it as secondary to the JC/IC operational surface. The design should decide whether it remains root-level or moves under a less prominent compatibility/maintenance grouping.
- Introduce a small CLI-layer `FlowNarrative` Pydantic model:
  ```python
  class FlowNarrative(BaseModel):
      model_config = ConfigDict(frozen=True)

      command_name: str
      purpose: str
      inputs: tuple[str, ...]
      steps: tuple[str, ...]
      outputs: tuple[str, ...]
  ```
- Use `FlowNarrative` instances to make CLI flows self-describing. The CLI should read like a project map: purpose, inputs, major steps, and outputs are visible near command registration.
- Make the evaluator -> writer -> pipeline-check sequence explicit in the `ic optimize` narrative while keeping the real orchestration inside existing runtime/state-machine modules.
- Stop requiring dataset/config/split/revision inputs for normal operator commands. Resolve the fixed project dataset identity internally through one project-specific helper/config.
- Keep existing request models as internal validation boundaries for now where practical, but stop exposing their dataset fields as public CLI inputs.
- Create a fresh session for every CLI invocation by default; keep `--session-id` as an explicit override.
- Remove public baseline-cache concepts, including `--invalidate-baseline-cache`.
- Replace operator-facing cache freshness control with `--fresh-worktrees`, scoped to refreshing selected repository worktrees.
- Keep only meaningful operator flags for JC/IC flows:
  - `--max-items`
  - `--offset`
  - `--max-workers`
  - `--projection-only`
  - `--yes`
  - `--session-id`
  - `--fresh-worktrees`
- Preserve Typer as the CLI framework and use nested Typer apps for natural `jc run`, `ic run`, and `ic optimize` command structure.
- Update tests and docs to describe the flow-oriented CLI rather than the historical baseline/experiment surface.

## Capabilities

### New Capabilities
- `flow-oriented-cli`: The CLI exposes JC and IC project flows with fixed dataset resolution, fresh sessions, worktree freshness controls, and clear flow narratives.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `entrypoints/cli.py` or a new `entrypoints/app.py` for Typer app and command registration.
  - A new module such as `entrypoints/flows.py` for `FlowNarrative` and static narrative definitions.
  - Request-building helpers in the CLI layer to resolve project dataset defaults internally.
  - Existing hosted/runtime/orchestration entrypoints that run localization evaluation and optimization.
  - `tests_harness/test_cli_guardrails.py` and any tests asserting old command names, dataset flags, baseline cache invalidation, or help output.
  - Developer docs such as `docs/lca_localization.md` and `docs/telemetry_sessions_cost.md`.
- Affected behavior:
  - The main public CLI becomes centered on `jc run`, `ic run`, and `ic optimize`.
  - Dataset identity is no longer required from normal operator CLI invocations.
  - Every command creates a fresh run session unless `--session-id` is supplied.
  - `--fresh-worktrees` replaces baseline-cache invalidation as the only public cache freshness control.
  - `ic optimize` help/source narrative makes the evaluator -> writer feedback -> candidate policy -> pipeline checks -> accept/reject flow obvious.
- Affected dependencies:
  - Typer should already be present after the previous CLI migration; verify the dependency remains present and do not add duplicate entries.
