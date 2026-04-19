## Why

`entrypoints/cli.py` currently mixes command declaration, argument parsing, compatibility logic, confirmation/projection flow, request construction, and orchestration dispatch in a way that makes the entrypoint hard to scan. Migrating the public CLI surface to Typer lets each command read like a normal typed Python function while preserving the existing Pydantic request models as the strict validation boundary.

## What Changes

- Replace the large argparse parser tree with a Typer root app exposing the existing public commands: `baseline`, `experiment`, and `sync-hf`.
- Complete the migration by removing argparse from the CLI implementation entirely: no argparse imports, parser construction, `_parse_args` compatibility helper, parser-specific tests, or argparse-only documentation should remain.
- Keep command functions thin: parse CLI parameters, construct typed request objects through helpers, and call existing orchestration/runtime modules.
- Preserve existing command names and flags where practical, including hosted dataset flags, session flags, projection-only flow, `--yes` automation guardrails, and baseline cache invalidation.
- Preserve removed-flag behavior for legacy flags that are intentionally unsupported by hosted Langfuse flows.
- Move request-building and shared command execution helpers out of the command declaration path where doing so makes the entrypoint easier to read.
- Update CLI tests to use Typer invocation utilities and to assert help output, command invocation, important validation behavior, removed-flag behavior, and exit-code compatibility.
- Add or retain the Typer dependency in `pyproject.toml` as the standardized CLI framework.

## Capabilities

### New Capabilities
- `typer-cli-surface`: The hosted localization CLI is exposed through Typer commands while retaining the existing typed request model boundary and command behavior.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `entrypoints/cli.py` and any new helper module such as `entrypoints/app.py` or `entrypoints/cli_helpers.py`.
  - `entrypoints/models/requests.py` only if small ergonomics changes are required; request model semantics should remain unchanged.
  - `tests_harness/test_cli_guardrails.py` and related CLI entrypoint tests.
  - Developer docs such as `docs/lca_localization.md` if command help or invocation examples need updates.
- Affected dependencies:
  - `pyproject.toml` must include Typer if it is not already present.
- Affected behavior:
  - `baseline`, `experiment`, and `sync-hf` remain the public command surface.
  - Root and command help should be available through Typer.
  - Existing request models remain the canonical validation boundary before orchestration.
  - argparse is removed as an implementation dependency for the CLI surface.
  - Interactive confirmation, projection-only, `--yes`, removed-flag handling, and exit codes are risk areas requiring explicit parity tests.
