## 1. Setup And Clean Replacement Plan

- [x] 1.1 Verify `pyproject.toml` includes a single Typer dependency entry and add `typer` if the dependency is missing in the active branch.
- [x] 1.2 Inventory the current argparse command surface in `entrypoints/cli.py`, including `baseline`, `experiment`, `sync-hf`, shared hosted flags, baseline-only flags, sync-only flags, and `REMOVED_FLAGS`.
- [x] 1.3 Decide the Typer-backed compatibility shape for `main(argv: list[str] | None)` so existing `run.py` and tests can invoke the Typer app without retaining argparse.
- [x] 1.4 Add Typer `app = typer.Typer(...)` wiring in `entrypoints/cli.py` or a small imported app module while keeping orchestration helpers available.

## 2. Shared CLI Helpers

- [x] 2.1 Extract or clarify shared hosted-run helpers for session construction, dataset fetch, deterministic selection, projection printing, confirmation, and Langfuse flushing without changing behavior.
- [x] 2.2 Add request-building helpers for `HostedLocalizationBaselineRequest` and `HostedLocalizationRunRequest` if they make the CLI flow easier to scan.
- [x] 2.3 Preserve removed-flag detection without argparse, before Typer's generic unknown-option handling, and keep the hosted-dataset compatibility message.
- [x] 2.4 Preserve stable monkeypatch targets used by existing tests or update tests to patch the new helper locations intentionally.

## 3. Migrate Commands

- [x] 3.1 Migrate `sync-hf` to a Typer command exposed as `sync-hf`, preserving `--hf-dataset`, required `--dataset`, required `--config`, required `--split`, `--revision`, and `--token`.
- [x] 3.2 Add `sync-hf` tests with `typer.testing.CliRunner` that verify required option behavior and argument forwarding to `sync_hf_localization_dataset_to_langfuse`.
- [x] 3.3 Migrate `baseline` to a thin Typer command preserving shared hosted flags and `--invalidate-baseline-cache`.
- [x] 3.4 Add baseline Typer tests covering request construction, `--projection-only`, `--yes`, `--yes` without `--max-items`, `--max-workers 0`, removed flags, and baseline cache invalidation.
- [x] 3.5 Migrate `experiment` to a thin Typer command preserving shared hosted flags and using the same hosted-run helper flow where practical.
- [x] 3.6 Add experiment Typer tests covering request construction, `--projection-only`, `--yes`, `--yes` without `--max-items`, `--max-workers 0`, and removed flags.

## 4. Help, Parity, And Cleanup

- [x] 4.1 Add Typer `CliRunner` tests for root `--help`, `baseline --help`, `experiment --help`, and `sync-hf --help`.
- [x] 4.2 Add or update parity tests for required options, non-zero exits, removed legacy flags, and public command names.
- [x] 4.3 Remove argparse parser construction, `_parse_args`, parser-only tests, argparse fallback paths, and all argparse imports.
- [x] 4.4 Search the CLI implementation, CLI tests, and developer docs for argparse artifacts and remove or rewrite anything tied to the old parser surface.
- [x] 4.5 Confirm command functions are materially shorter and easier to scan than the old argparse surface, with business logic kept in helpers or existing hosted/runtime modules.
- [x] 4.6 Update developer docs or examples such as `docs/lca_localization.md` only where command help, invocation examples, or testing guidance changed.

## 5. Verification

- [x] 5.1 Run the migrated CLI guardrail tests, including `tests_harness/test_cli_guardrails.py`.
- [x] 5.2 Run relevant hosted baseline/experiment tests that exercise request models and orchestration boundaries.
- [x] 5.3 Run static checks used by the repo for touched CLI and test files.
- [x] 5.4 Manually verify root and per-command help output through the Typer app.
- [x] 5.5 Confirm no new public commands or hosted-run semantics were introduced beyond parity.
- [x] 5.6 Confirm the final diff contains no argparse implementation artifacts for the CLI surface.
