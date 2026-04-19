## Context

The current CLI entrypoint is `entrypoints/cli.py`. It uses argparse to declare the command tree and also contains removed-flag compatibility checks, hosted dataset selection, cost projection, interactive confirmation, request model construction, baseline cache invalidation, orchestration dispatch, result printing, error handling, and Langfuse flushing.

The hosted command boundary is already typed through Pydantic request models in `entrypoints/models/requests.py`, especially `HostedLocalizationBaselineRequest` and `HostedLocalizationRunRequest`. The migration should make the entrypoint easier to read without weakening that boundary or moving hosted experiment semantics into the CLI layer.

Typer is the chosen CLI framework because its typed command functions match the existing request-model style, command declarations are easier to scan than a large parser tree, help and subcommand help are first-class, and `typer.testing.CliRunner` gives ergonomic command invocation tests.

## Goals / Non-Goals

**Goals:**

- Expose `baseline`, `experiment`, and `sync-hf` through a Typer root app.
- Preserve existing command names and flags where practical.
- Keep command functions thin and readable:
  1. CLI parameters
  2. typed request construction
  3. orchestration/helper call
- Keep `HostedLocalizationBaselineRequest` and `HostedLocalizationRunRequest` as the canonical validation boundary for hosted baseline and experiment requests.
- Keep orchestration in existing hosted/runtime modules such as `telemetry.hosted.experiments` and `telemetry.hosted.hf_lca`.
- Preserve automation behavior, guardrails, projection-only behavior, interactive confirmation behavior, and removed-flag errors.
- Make root help and per-command help available through Typer.
- Update tests to cover Typer command invocation, key flags, validation, removed-flag behavior, and exit codes.
- Finish with no argparse artifacts in the CLI implementation, tests, or docs for this surface.

**Non-Goals:**

- Do not redesign hosted experiment or hosted baseline semantics.
- Do not change request model semantics unless a narrow CLI ergonomics issue requires it.
- Do not merge orchestration into Typer command bodies.
- Do not broaden the public command surface beyond parity needs.
- Do not add new hosted localization features as part of the migration.
- Do not keep argparse as a fallback parser, compatibility layer, hidden alternate entrypoint, or long-term test target.

## Decisions

### Use a clean in-place replacement

Replace argparse in the existing CLI surface with Typer. The final state is a clean Typer CLI with no argparse parser, imports, `_parse_args` helper, parser-only tests, or argparse-specific documentation for this command surface.

The implementation can still be reviewed in small commits, but those commits should move monotonically toward deletion of argparse. "Staged" means command-by-command review and parity checks, not long-term coexistence and not a fallback implementation.

1. Add a Typer `app = typer.Typer(...)` root while keeping current execution helpers callable.
2. Migrate one command at a time behind the same command names.
3. Update tests for each migrated command with `CliRunner`.
4. Remove argparse parser construction, `_parse_args`, argparse imports, and dead parser-only tests before the change is considered complete.

This avoids maintaining two public command surfaces while still letting review happen command by command. If temporary adapter functions are useful during implementation, they must delegate to Typer and must not retain argparse behavior. Any temporary adapter exists only to preserve the current `main(argv: list[str] | None)` and `run.py` call pattern, and it must remain in the final code only if it is Typer-backed and contains no argparse logic.

Alternative considered: introduce a parallel `typer-cli` entrypoint and switch callers later. That is rejected for this change because it risks drift between argparse and Typer behavior, delays the readability benefit in `entrypoints/cli.py`, and makes it too easy to leave argparse artifacts behind.

### Keep `entrypoints/cli.py` as the public compatibility module

`entrypoints/cli.py` should continue to expose the public module-level entrypoint used by `run.py` and tests. It may either own the Typer app directly or import it from a small `entrypoints/app.py` module. The preferred shape is:

- `entrypoints/cli.py` owns `app = typer.Typer(...)` and `main(...)` for public compatibility.
- Thin Typer command functions live near the app declaration.
- Larger execution helpers remain below the command declaration or move to a helper module if that materially improves readability.

If the file remains hard to scan after command migration, split helper code into a private module such as `entrypoints/cli_helpers.py`. The split should be mechanical and should not change orchestration behavior.

The final module must not import argparse and must not expose `_parse_args` as an argparse compatibility API. Tests that currently call `_parse_args` should be rewritten to invoke the Typer app through `CliRunner` or to exercise Typer-backed compatibility entrypoints.

### Use one thin command function per public command

Target command functions:

- `baseline(...)`
- `experiment(...)`
- `sync_hf(...)`, exposed as `sync-hf`

The hosted `baseline` and `experiment` signatures should keep the existing shared flags:

- `--dataset` required
- `--version`
- `--max-items`
- `--offset`
- `--max-workers`
- `--projection-only`
- `--yes`
- `--session-id`
- `--allow-session-fallback`

`baseline` should also keep `--invalidate-baseline-cache`.

`sync-hf` should keep:

- `--hf-dataset`
- `--dataset` required
- `--config` required
- `--split` required
- `--revision`
- `--token`

Typer type hints should represent CLI parsing types only. Business validation remains in request models and existing helpers. For example, Typer can parse `max_workers: int`, but positive-worker semantics should continue to be enforced by the Pydantic request model and existing guardrail behavior.

### Preserve the typed request boundary through request-building helpers

Introduce helper functions if useful:

- `build_baseline_request(...) -> HostedLocalizationBaselineRequest`
- `build_experiment_request(...) -> HostedLocalizationRunRequest`
- shared helpers for session config, dataset selection, projection, and confirmation

The helpers should make it obvious where CLI arguments become request models. They should not duplicate Pydantic field validation or move orchestration concerns into the CLI declaration path.

### Keep confirmation, projection-only, and `--yes` semantics compatible

The hosted baseline and experiment flow must preserve the current ordering:

1. Fetch hosted dataset.
2. Select tasks deterministically from `--offset` and `--max-items`.
3. Print selection and projection.
4. Enforce `--yes` requiring `--max-items`.
5. Require interactive confirmation unless `--yes` is set.
6. Exit before orchestration when `--projection-only` is set.
7. Build the typed request object.
8. Call the existing orchestration function.
9. Flush Langfuse in the same finalization path.

Interactive confirmation is a risk area because Typer and `CliRunner` alter stdin/stdout handling. Tests should cover non-interactive failure, `--yes` bypass, and projection-only not invoking orchestration.

### Preserve removed-flag behavior explicitly

The current parser rejects legacy flags such as `--mode`, `--dataset-source`, and pricing assumption flags with a hosted-dataset-specific error message. Typer will otherwise produce a generic unknown-option error, so the migration should preserve the intentional removed-flag message.

Implement this without argparse. Prefer a small raw-argv compatibility scan before Typer invocation or a Typer callback that preserves the message and exit behavior. The removed-flag check is compatibility logic, not a reason to keep argparse.

### Testing strategy

Use `typer.testing.CliRunner` for root and command invocation tests:

- Root `--help` exits successfully.
- `baseline --help`, `experiment --help`, and `sync-hf --help` exit successfully.
- Required options produce non-zero exits.
- Removed flags produce the existing hosted-dataset compatibility message.
- `baseline` and `experiment` construct the expected request models and call the current hosted orchestration functions when guarded with `--yes --max-items`.
- `--projection-only` exits before orchestration.
- `--yes` without `--max-items` remains rejected.
- `--max-workers 0` remains rejected by the existing validation path.
- `sync-hf` passes Hugging Face and Langfuse arguments through to the existing sync helper.

Existing tests in `tests_harness/test_cli_guardrails.py` should be migrated rather than deleted unless their assertion is parser-specific and replaced by an equivalent Typer behavior assertion.

Tests that directly assert argparse internals or `_parse_args` behavior should be removed or rewritten to assert equivalent public Typer CLI behavior. The test suite should not retain argparse as a supported implementation detail.

### Dependency handling

`pyproject.toml` already contains `typer>=0.24.1` in this workspace. The implementation task should verify that dependency remains present and no duplicate dependency entry is introduced. If another branch lacks Typer, add it there as part of the migration.

## Risks / Trade-offs

- [Risk] Typer's default unknown-option and validation messages differ from argparse. → Mitigation: add parity tests for removed flags, required options, and important validation failures; preserve custom removed-flag scanning where needed.
- [Risk] Exit codes may change for parser errors or callback exceptions. → Mitigation: assert expected success/non-zero outcomes with `CliRunner` and keep current command-level guardrail semantics stable.
- [Risk] Interactive confirmation behavior can regress under non-TTY automation. → Mitigation: keep confirmation logic separate from Typer command declaration and test `--yes`, projection-only, and non-interactive paths.
- [Risk] Command functions can accumulate business logic during migration. → Mitigation: keep command bodies as thin adapters and move execution/request-building into helpers.
- [Risk] Splitting `entrypoints/cli.py` can break monkeypatch targets in tests. → Mitigation: preserve public imports or add compatibility wrappers for stable test targets while migrating.
- [Risk] Typer option naming for underscores differs from existing flag names. → Mitigation: explicitly set option declarations where needed so public flags remain `--max-items`, `--projection-only`, `--session-id`, `--allow-session-fallback`, and `--invalidate-baseline-cache`.
- [Risk] Temporary migration scaffolding can become permanent and leave argparse artifacts behind. → Mitigation: make argparse deletion an explicit acceptance criterion and verification task; the change is not done until `entrypoints/cli.py` and CLI tests contain no argparse dependency.

## Migration Plan

1. Verify Typer is present in `pyproject.toml`.
2. Add a Typer root app and compatibility `main(argv)` wrapper in `entrypoints/cli.py`.
3. Migrate `sync-hf` first because it has the smallest execution flow.
4. Migrate `baseline` with parity tests for projection, confirmation, request construction, and cache invalidation.
5. Migrate `experiment` using shared hosted-run helpers from the baseline path where practical.
6. Remove argparse parser construction, `_parse_args`, argparse imports, parser-specific tests, and argparse-only documentation before declaring the migration complete.
7. Update developer docs/help examples if any wording or invocation guidance changes.

Rollback during implementation should use version control to return to the previous argparse implementation if Typer parity cannot be achieved in the branch. The merged final state must not contain both implementations.

## Open Questions

- Should `main(argv)` return Typer's result directly or continue to present a `None`-returning compatibility surface for existing tests and callers?
- Which current argparse error messages, beyond removed-flag handling, need exact text parity versus non-zero behavior parity?
