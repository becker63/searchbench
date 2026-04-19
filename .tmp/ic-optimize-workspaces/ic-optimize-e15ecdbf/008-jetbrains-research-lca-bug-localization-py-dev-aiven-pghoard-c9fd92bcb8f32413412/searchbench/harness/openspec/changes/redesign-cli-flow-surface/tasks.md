## 1. Flow Narrative Foundation

- [x] 1.1 Verify `pyproject.toml` already includes Typer and avoid adding duplicate dependency entries.
- [x] 1.2 Add `entrypoints/flows.py` with frozen Pydantic `FlowNarrative` model containing `command_name`, `purpose`, `inputs`, `steps`, and `outputs`.
- [x] 1.3 Define `FlowNarrative` constants for `jc run`, `ic run`, and `ic optimize`.
- [x] 1.4 Include evaluator -> writer feedback -> candidate policy -> pipeline checks -> accept/reject steps in the `ic optimize` narrative.
- [x] 1.5 Add tests that verify all primary flow commands have matching narratives and the `ic optimize` narrative includes evaluator, writer, candidate policy, pipeline, and accept/reject concepts.

## 2. Fixed Dataset And Shared Flow Helpers

- [x] 2.1 Add a single project dataset resolution helper/config for normal JC and IC flows.
- [x] 2.2 Update request-building helpers so normal flow commands populate dataset identity internally instead of requiring public dataset flags.
- [x] 2.3 Keep existing request models as internal validation boundaries unless a narrow adapter is needed for CLI ergonomics.
- [x] 2.4 Ensure every flow command creates a fresh run-scope session by default.
- [x] 2.5 Keep `--session-id` as an explicit override and remove public `allow_session_fallback` exposure from normal flow commands.

## 3. Typer Command Tree

- [x] 3.1 Introduce nested Typer sub-apps for `jc` and `ic` under the existing root app.
- [x] 3.2 Add thin `jc run` command registration with meaningful operator flags and `FlowNarrative`-driven help or summary text.
- [x] 3.3 Add thin `ic run` command registration with meaningful operator flags and `FlowNarrative`-driven help or summary text.
- [x] 3.4 Add thin `ic optimize` command registration with meaningful operator flags and `FlowNarrative`-driven help or summary text.
- [x] 3.5 Keep `sync-hf` available for ingestion while keeping it secondary to the JC/IC flow commands.
- [x] 3.6 Add `CliRunner` help tests for root help, `jc run --help`, `ic run --help`, `ic optimize --help`, and `sync-hf --help`.

## 4. Execution Delegation

- [x] 4.1 Implement JC run executor/helper that creates a session, loads the fixed task set, optionally refreshes worktrees, runs the JC localizer/evaluator flow, and reports results.
- [x] 4.2 Implement IC run executor/helper that creates a session, loads the fixed task set, optionally refreshes worktrees, runs the IC localizer/evaluator flow, and reports results.
- [x] 4.3 Implement IC optimize executor/helper that delegates to existing optimization orchestration such as `orchestration.machine.run_loop(...)` or a narrow adapter around it.
- [x] 4.4 Ensure `ic optimize` does not duplicate `build_loop_dependencies`, `build_iteration_feedback`, `attempt_policy_generation`, pipeline classification, or state-machine transition logic in the CLI.
- [x] 4.5 Add command invocation tests proving JC/IC commands delegate to the expected runtime/orchestration helpers without requiring real network calls.

## 5. Cache And Legacy Surface Cleanup

- [x] 5.1 Replace public `--invalidate-baseline-cache` behavior with `--fresh-worktrees` for JC and IC flow commands.
- [x] 5.2 Implement `--fresh-worktrees` so it refreshes or invalidates selected repository worktrees only.
- [x] 5.3 Add tests proving `--fresh-worktrees` refreshes selected worktrees and does not touch baseline bundle cache paths.
- [x] 5.4 Remove baseline-cache terminology from CLI help, command bodies, and developer docs for the public operator surface.
- [x] 5.5 Remove old `baseline` and `experiment` commands entirely, with no aliases, hidden commands, or compatibility error commands.
- [x] 5.6 Add tests for old command/flag behavior, including removed dataset-oriented flags and `--invalidate-baseline-cache`.

## 6. Operator Flag Simplification

- [x] 6.1 Remove public `--dataset`, `--config`, `--split`, and `--revision` requirements from normal JC/IC commands.
- [x] 6.2 Preserve meaningful operator flags where applicable: `--max-items`, `--offset`, `--max-workers`, `--projection-only`, `--yes`, `--session-id`, and `--fresh-worktrees`.
- [x] 6.3 Add tests proving normal JC/IC flow commands do not require dataset/config/split/revision inputs.
- [x] 6.4 Add tests proving repeated invocations produce fresh sessions by default and `--session-id` overrides the generated session.

## 7. Docs And Verification

- [x] 7.1 Update CLI/developer docs such as `docs/lca_localization.md` to present `jc run`, `ic run`, and `ic optimize` as the primary flows.
- [x] 7.2 Update telemetry/session docs if public session semantics or command examples change.
- [x] 7.3 Run CLI guardrail tests and new flow narrative tests.
- [x] 7.4 Run relevant hosted/runtime/orchestration regression tests touched by the command executors.
- [x] 7.5 Run static checks for touched entrypoint, flow, test, and docs files.
- [x] 7.6 Manually verify help output for root, `jc run`, `ic run`, `ic optimize`, and `sync-hf`.
