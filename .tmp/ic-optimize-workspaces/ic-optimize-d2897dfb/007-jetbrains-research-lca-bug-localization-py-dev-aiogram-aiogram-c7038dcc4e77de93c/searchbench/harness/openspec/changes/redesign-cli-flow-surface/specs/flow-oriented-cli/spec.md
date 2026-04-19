## ADDED Requirements

### Requirement: Flow-oriented command tree
The system SHALL expose the main operator CLI as flow-oriented Typer commands centered on JC and IC operations.

#### Scenario: Root help advertises project flows
- **WHEN** a user invokes root CLI help
- **THEN** the help lists JC and IC command groups as the primary operator surface

#### Scenario: JC run command is available
- **WHEN** a user invokes help for `jc run`
- **THEN** the CLI exits successfully and describes the JC run flow

#### Scenario: IC run command is available
- **WHEN** a user invokes help for `ic run`
- **THEN** the CLI exits successfully and describes the IC run flow

#### Scenario: IC optimize command is available
- **WHEN** a user invokes help for `ic optimize`
- **THEN** the CLI exits successfully and describes the IC optimization flow

### Requirement: Flow narratives describe operational architecture
The system SHALL define a CLI-layer Pydantic `FlowNarrative` model with static narratives for `jc run`, `ic run`, and `ic optimize`.

#### Scenario: Flow narrative shape exists
- **WHEN** maintainers inspect the CLI flow narrative module
- **THEN** it contains a frozen Pydantic `FlowNarrative` model with command name, purpose, inputs, steps, and outputs

#### Scenario: Narratives align with commands
- **WHEN** tests compare registered flow commands with narrative definitions
- **THEN** every primary flow command has a matching `FlowNarrative`

#### Scenario: IC optimization narrative shows handoff
- **WHEN** maintainers inspect the `ic optimize` narrative
- **THEN** its steps explicitly include evaluation, writer feedback construction, candidate policy generation, pipeline correctness checks, and accept/reject reporting

### Requirement: Fixed project dataset resolution
The system SHALL resolve normal JC and IC run dataset identity internally instead of requiring dataset/config/split/revision CLI arguments.

#### Scenario: Normal run does not require dataset input
- **WHEN** a user invokes `jc run`, `ic run`, or `ic optimize` with required safety flags
- **THEN** the command does not require `--dataset`, `--config`, `--split`, or `--revision`

#### Scenario: Dataset identity is centralized
- **WHEN** command helpers build internal request objects or load tasks
- **THEN** they obtain dataset identity from one project-specific helper or config object

### Requirement: Fresh session defaults
The system SHALL create a fresh session for every CLI invocation by default and allow explicit override with `--session-id`.

#### Scenario: Fresh sessions by default
- **WHEN** the same flow command is invoked twice without `--session-id`
- **THEN** each invocation receives a distinct run-scope session id

#### Scenario: Explicit session override
- **WHEN** a user invokes a flow command with `--session-id`
- **THEN** the command uses that session id for the run

### Requirement: Worktree freshness replaces baseline cache controls
The system SHALL expose `--fresh-worktrees` as the public cache freshness control for flow commands and SHALL NOT expose baseline-cache invalidation in the new public CLI.

#### Scenario: Fresh worktrees refreshes selected worktrees
- **WHEN** a user invokes a flow command with `--fresh-worktrees`
- **THEN** the command refreshes or invalidates selected repository worktrees before execution

#### Scenario: Baseline cache flag is removed
- **WHEN** a user invokes a new flow command with `--invalidate-baseline-cache`
- **THEN** the CLI rejects the flag and does not expose baseline bundle cache invalidation as a public operator control

### Requirement: Legacy command behavior is explicit
The system SHALL remove old `baseline` and `experiment` commands entirely from the public CLI.

#### Scenario: Baseline command is absent
- **WHEN** a user invokes `baseline`
- **THEN** the CLI treats it as an unknown command because no public command, alias, hidden command, or compatibility shim is registered

#### Scenario: Experiment command is absent
- **WHEN** a user invokes `experiment`
- **THEN** the CLI treats it as an unknown command because no public command, alias, hidden command, or compatibility shim is registered

### Requirement: Thin command delegation
The system SHALL keep Typer command bodies thin and delegate execution to existing runtime/orchestration helpers.

#### Scenario: IC optimization delegates orchestration
- **WHEN** `ic optimize` executes
- **THEN** the command delegates the optimization loop to existing orchestration/runtime code rather than duplicating state-machine, feedback, writer, or pipeline logic in the CLI body

#### Scenario: Commands preserve operator flags
- **WHEN** users invoke flow commands
- **THEN** meaningful operator flags such as `--max-items`, `--offset`, `--max-workers`, `--projection-only`, `--yes`, `--session-id`, and `--fresh-worktrees` remain available where applicable
