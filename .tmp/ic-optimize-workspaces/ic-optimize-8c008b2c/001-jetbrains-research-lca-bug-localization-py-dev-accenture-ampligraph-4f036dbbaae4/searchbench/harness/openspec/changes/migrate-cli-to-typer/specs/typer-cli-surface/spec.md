## ADDED Requirements

### Requirement: Typer root command surface
The system SHALL expose the hosted localization CLI through a Typer root application with the existing public commands `baseline`, `experiment`, and `sync-hf`, with no argparse implementation artifacts remaining for this CLI surface.

#### Scenario: Root help is available
- **WHEN** a user invokes the CLI with `--help`
- **THEN** the CLI exits successfully and displays the Typer root help including the hosted localization commands

#### Scenario: Command help is available
- **WHEN** a user invokes `baseline --help`, `experiment --help`, or `sync-hf --help`
- **THEN** the CLI exits successfully and displays help for that command

#### Scenario: Argparse implementation is removed
- **WHEN** the migration is complete
- **THEN** the CLI implementation contains no argparse imports, argparse parser construction, `_parse_args` argparse compatibility helper, argparse fallback command path, or parser-specific tests

### Requirement: Hosted request models remain canonical validation
The system SHALL continue constructing `HostedLocalizationBaselineRequest` and `HostedLocalizationRunRequest` before entering hosted baseline or experiment orchestration.

#### Scenario: Baseline constructs typed request
- **WHEN** a guarded baseline invocation passes CLI parsing and confirmation requirements
- **THEN** the CLI constructs a `HostedLocalizationBaselineRequest` with the selected dataset, version, session, selection, projection, limit, offset, and worker arguments before calling baseline orchestration

#### Scenario: Experiment constructs typed request
- **WHEN** a guarded experiment invocation passes CLI parsing and confirmation requirements
- **THEN** the CLI constructs a `HostedLocalizationRunRequest` with the selected dataset, version, session, selection, projection, limit, offset, and worker arguments before calling experiment orchestration

#### Scenario: Request validation remains active
- **WHEN** a hosted command receives values rejected by the request model, such as a non-positive worker count
- **THEN** the command fails before orchestration using the existing validation boundary

### Requirement: Thin command bodies
The system SHALL keep Typer command functions focused on CLI parameters, request construction, and orchestration/helper calls, without embedding hosted business logic in the command declarations.

#### Scenario: Baseline entrypoint remains readable
- **WHEN** maintainers inspect the baseline command implementation
- **THEN** the flow from CLI parameters to request construction to orchestration call is apparent without reading a large parser tree

#### Scenario: Experiment entrypoint remains readable
- **WHEN** maintainers inspect the experiment command implementation
- **THEN** the flow from CLI parameters to request construction to orchestration call is apparent without reading a large parser tree

### Requirement: Existing hosted command behavior is preserved
The system SHALL preserve hosted CLI behavior for command names, practical public flags, projection-only flow, interactive confirmation flow, `--yes` automation guardrails, baseline cache invalidation, and Hugging Face sync argument forwarding.

#### Scenario: Projection-only exits before orchestration
- **WHEN** a user invokes `baseline` or `experiment` with `--projection-only`
- **THEN** the CLI computes and prints selection/projection information and exits before calling hosted orchestration

#### Scenario: Yes guardrail is enforced
- **WHEN** a user invokes `baseline` or `experiment` with `--yes` and without `--max-items`
- **THEN** the CLI rejects the invocation before orchestration

#### Scenario: Sync forwards arguments
- **WHEN** a user invokes `sync-hf` with dataset, config, split, and optional revision/token arguments
- **THEN** the CLI forwards those values to the existing Hugging Face to Langfuse sync helper

#### Scenario: Baseline cache invalidation remains available
- **WHEN** a user invokes `baseline` with `--invalidate-baseline-cache`
- **THEN** the CLI preserves the current baseline bundle and selected worktree cache invalidation behavior before running baseline orchestration

### Requirement: Removed legacy flags remain rejected intentionally
The system SHALL preserve explicit rejection of legacy removed flags with the hosted-dataset compatibility message instead of silently accepting them or replacing the behavior with an unhelpful generic parser failure.

#### Scenario: Removed flag fails with compatibility guidance
- **WHEN** a user invokes a hosted command with a removed flag such as `--mode`, `--dataset-source`, or `--assume-input-tokens`
- **THEN** the CLI exits non-zero and reports that the flag is no longer supported for hosted Langfuse dataset runs

### Requirement: Typer invocation tests cover CLI parity
The system SHALL test the Typer CLI with `typer.testing.CliRunner` for command invocation and important flag/validation behavior.

#### Scenario: Command invocation tests use Typer runner
- **WHEN** CLI guardrail tests invoke the root app or a command
- **THEN** they use Typer testing utilities and assert exit status, output, and orchestration calls without requiring real Langfuse or Hugging Face network access

#### Scenario: Public surface is not broadened
- **WHEN** CLI parity tests inspect or invoke available commands
- **THEN** only the intended public commands needed for parity are exposed
