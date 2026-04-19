## Context

The current Typer CLI in `entrypoints/cli.py` still exposes older hosted localization nouns: `baseline`, `experiment`, and `sync-hf`. It also keeps dataset-oriented public flags and baseline cache invalidation even though this project now has a fixed dataset identity and the operator-facing flow is better described as:

- run JC
- run IC
- optimize IC

The optimization architecture already exists below the CLI boundary. `orchestration.machine.build_loop_dependencies()` wires evaluation, feedback construction, writer policy generation, and pipeline execution through `build_loop_dependencies(...)`, `build_iteration_feedback(...)`, and `attempt_policy_generation(...)`. `orchestration.machine.run_loop(...)` then runs the closed-loop optimizer. The CLI redesign should narrate that architecture clearly without reimplementing the state machine or turning CLI command functions into orchestration code.

The previous migration standardized the CLI on Typer and removed argparse. This change is not another framework migration. It is a CLI architecture change: the entrypoint should become the clearest high-level map of the project’s operational flows.

## Goals / Non-Goals

**Goals:**

- Replace the primary public CLI surface with flow-oriented commands:
  - `jc run`
  - `ic run`
  - `ic optimize`
- Keep `sync-hf` available for ingestion during this change, but keep it visually secondary to the JC/IC flow surface.
- Add `FlowNarrative` in the CLI layer with static definitions for the new flows.
- Make `ic optimize` visibly narrate evaluator -> writer feedback -> candidate policy -> pipeline checks -> accept/reject.
- Resolve fixed project dataset identity internally instead of requiring `--dataset` for normal operator commands.
- Create a fresh session for every invocation by default while retaining `--session-id` override.
- Remove public baseline-cache terminology and `--invalidate-baseline-cache`.
- Expose worktree freshness as `--fresh-worktrees`.
- Keep command bodies thin and delegate to existing hosted/runtime/orchestration helpers.
- Preserve Typer as the CLI framework and use `CliRunner` tests.

**Non-Goals:**

- Do not redesign the internal optimization state machine.
- Do not rewrite localization evaluation, writer generation, pipeline execution, or finalization logic.
- Do not broaden the public operator surface beyond JC/IC/optimization flows and ingestion support.
- Do not keep dataset/config/split/revision exposed in normal operator commands unless a compatibility path is explicitly chosen.
- Do not preserve baseline-cache concepts in the new public CLI.
- Do not turn `FlowNarrative` into a runtime engine, planner, state machine, or source of execution truth.

## Decisions

### Use `entrypoints/cli.py` for Typer registration and `entrypoints/flows.py` for narratives

Target module structure:

- `entrypoints/cli.py`
  - owns the root `typer.Typer(...)` app
  - registers `jc_app`, `ic_app`, and the `sync-hf` command
  - contains thin command functions
  - delegates to helper/executor functions for selection, projection, session, worktree refresh, and orchestration calls
- `entrypoints/flows.py`
  - defines `FlowNarrative`
  - defines static narrative constants for `jc run`, `ic run`, and `ic optimize`
  - optionally exposes lookup helpers such as `get_flow_narrative(command_name: str)`
- `entrypoints/project_dataset.py` or a small helper inside the CLI layer
  - resolves the fixed project dataset name/version/config/split from constants or environment-backed config
  - returns a typed value used to populate existing request models
- Existing modules under `telemetry.hosted`, `localization.runtime`, and `orchestration` remain the execution owners.

This split keeps the CLI readable as a map while avoiding a large monolithic entrypoint. If implementation shows the helper module would be too small, dataset resolution can initially live near other CLI helpers, but it should still be a single explicit helper.

### Use nested Typer apps for the command tree

Target command tree:

- `jc run`
- `ic run`
- `ic optimize`
- `sync-hf`

`jc` and `ic` should be Typer sub-apps because nested commands make backend identity and loop type clear. `sync-hf` can remain root-level in this change to avoid hiding ingestion tooling before docs/tests are updated. The design should keep it secondary in help ordering/copy and avoid expanding it into a peer flow. A later change can move ingestion under `dataset sync-hf` or similar if the project wants stricter separation.

Old commands must be removed from the public CLI entirely. `baseline` and `experiment` should not be registered as commands, aliases, hidden commands, or compatibility error shims. Tests should assert they are absent from help and unavailable as public commands.

### Add `FlowNarrative` as a small Pydantic descriptive model

Use this shape unless implementation reveals a concrete typing issue:

```python
from pydantic import BaseModel, ConfigDict


class FlowNarrative(BaseModel):
    model_config = ConfigDict(frozen=True)

    command_name: str
    purpose: str
    inputs: tuple[str, ...]
    steps: tuple[str, ...]
    outputs: tuple[str, ...]
```

Narratives should be static, human-readable Pydantic model instances close to command registration. Pydantic keeps this consistent with the repo's existing typed-boundary style and makes future validation/serialization straightforward if help text, summaries, tests, or logs consume the narratives. Those uses are secondary. The important requirement is that a reader can open the CLI/flows module and understand what each flow does without reverse-engineering the state machine.

Intended narratives:

- `jc run`
  - create session
  - load fixed task set
  - optionally refresh worktrees
  - run JC localizer/evaluator flow
  - report results
- `ic run`
  - create session
  - load fixed task set
  - optionally refresh worktrees
  - run IC localizer/evaluator flow
  - report results
- `ic optimize`
  - create session
  - load fixed task set
  - optionally refresh worktrees
  - evaluate current IC behavior
  - build writer feedback from evaluation
  - generate candidate policy update
  - run pipeline correctness checks
  - accept or reject the update
  - report iteration outcome

Alternative considered: document this only in docs. That is insufficient because the CLI should be the operator-facing map and tests should be able to assert that command registration and narrative definitions stay aligned.

### Represent evaluator -> writer handoff narratively, not operationally

`ic optimize` should make the evaluator -> writer handoff explicit in the CLI source through `FlowNarrative.steps` and, optionally, command help/pre-run summary text. The actual handoff remains in orchestration:

- `build_loop_dependencies(...)` assembles runtime collaborators.
- `build_iteration_feedback(...)` turns evaluation results into writer-facing feedback.
- `attempt_policy_generation(...)` calls the writer with the optimization brief and pipeline feedback.
- pipeline helpers classify correctness checks and finalization helpers accept/reject the candidate policy.

The CLI should call an optimization executor that delegates to `orchestration.machine.run_loop(...)` or a narrow adapter around it. It should not duplicate dependency wiring, writer prompt assembly, pipeline classification, or state transitions.

### Keep current request models as internal boundaries for now

Existing request models such as `HostedLocalizationBaselineRequest` and `HostedLocalizationRunRequest` can remain internal validation boundaries during this redesign, even if their names become less ideal. The CLI should stop exposing dataset arguments and should populate request fields from a fixed project dataset helper.

If old request names become misleading after the CLI surface changes, propose a later cleanup to rename or replace them. This change should prioritize public CLI clarity without forcing a broad model rename.

### Resolve fixed project dataset identity in one place

Normal operator commands should not require `--dataset`, `--config`, `--split`, or `--revision`. Add one helper/config object that resolves project dataset identity and any default version needed by hosted loaders. This helper should be easy to test and should be the only place normal CLI flows know dataset identity.

`sync-hf` may still need ingestion-specific dataset/config/split/revision controls because ingestion is not a normal run/optimization flow. The proposal keeps `sync-hf` as a special case for now.

### Session and cache semantics

Every command should create a fresh run session by default using the existing session policy machinery. `--session-id` remains for explicit override. `allow_session_fallback` should not remain a prominent public option; if still needed internally, set it in the command helper/request construction path.

`--fresh-worktrees` is the only public freshness control for JC/IC flows. It should invalidate or refresh selected repository worktrees before execution. It must not mention or remove baseline bundles, and it must not expose `HARNESS_BASELINE_CACHE_DIR` behavior as an operator concept.

### Testing strategy

Use Typer `CliRunner` for command tests:

- root help includes `jc`, `ic`, and `sync-hf`
- `jc run --help`, `ic run --help`, and `ic optimize --help` are available
- command help or command metadata includes the associated `FlowNarrative` purpose/steps
- normal JC/IC commands do not require dataset flags
- old `baseline` and `experiment` commands are absent from public help and unavailable as registered commands
- `--invalidate-baseline-cache` is rejected and replaced by `--fresh-worktrees`
- `--fresh-worktrees` triggers selected worktree refresh and does not touch baseline bundle cache paths
- each invocation gets a fresh session by default
- `--session-id` overrides the generated session
- `FlowNarrative` definitions stay aligned with registered commands
- `ic optimize` tests verify the command delegates to optimization orchestration and its narrative contains evaluator, writer feedback, candidate policy, pipeline checks, and accept/reject steps

## Risks / Trade-offs

- [Risk] Old automation may still call `baseline` or `experiment`. -> Mitigation: remove the commands deliberately, update docs/tests, and let Typer report them as unknown commands rather than keeping compatibility shims.
- [Risk] Request model names remain historically loaded. -> Mitigation: keep them internal for now and document a later cleanup rather than widening this change.
- [Risk] `FlowNarrative` could grow into a second orchestration framework. -> Mitigation: keep it as a frozen Pydantic `BaseModel`, static, and descriptive only; execution continues through existing helpers.
- [Risk] Fixed dataset resolution can hide useful debugging knobs. -> Mitigation: centralize the helper and make overrides deliberate internal config/env behavior rather than public command flags.
- [Risk] `--fresh-worktrees` can be mistaken for all cache invalidation. -> Mitigation: help text and tests must scope it to selected worktree refresh only.
- [Risk] IC optimization source can still be hard to follow if command bodies only call a helper. -> Mitigation: keep the `FlowNarrative` visible near command registration and test that the narrative covers evaluator -> writer -> pipeline-check sequence.

## Migration Plan

1. Add `entrypoints/flows.py` with `FlowNarrative` and static narratives for `jc run`, `ic run`, and `ic optimize`.
2. Add fixed project dataset resolution helper/config for normal operator flows.
3. Introduce Typer sub-apps for `jc` and `ic` while keeping `sync-hf`.
4. Add thin command functions for `jc run`, `ic run`, and `ic optimize`.
5. Add executor/request-building helpers that populate fixed dataset identity, create sessions, select tasks, handle projection, and delegate to current runtime/orchestration modules.
6. Replace `--invalidate-baseline-cache` public behavior with `--fresh-worktrees`.
7. Remove old `baseline` and `experiment` commands entirely and remove dataset-oriented public flags for normal flows.
8. Update tests to assert the new command tree, narratives, session defaults, removed behavior, and worktree refresh behavior.
9. Update docs to present JC/IC/optimization flows as the primary CLI.

Rollback should be version-control based before merging. The final merged state should not preserve the historical baseline/experiment surface as the main operator CLI.

## Open Questions

- Should `sync-hf` remain root-level permanently, or should a later change move ingestion under a dedicated dataset/maintenance group?
- What exact constants/env names should define the fixed project dataset identity?
- None.
