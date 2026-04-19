## ADDED Requirements

### Requirement: Observation start compatibility
The system SHALL start Langfuse observations via a wrapper that accepts both context-manager objects and plain objects exposing `end()` without requiring `__enter__`/`__exit__`.

#### Scenario: Context manager observation
- **WHEN** the Langfuse client returns a context-manager observation object
- **THEN** the wrapper SHALL enter/exit it correctly and yield the observation handle to callers

#### Scenario: Plain object observation
- **WHEN** the Langfuse client returns a plain observation object exposing `end()` but no context manager methods
- **THEN** the wrapper SHALL yield the object and ensure `end()` is invoked exactly once on exit

### Requirement: Observation shutdown compatibility
The system SHALL end observations through a centralized helper that calls `update(...)` for metadata/output when available and calls `end()` without passing unsupported kwargs.

#### Scenario: End without error kwarg support
- **WHEN** an observation handle lacks support for `end(error=...)`
- **THEN** the helper SHALL record the error via `update(...)` metadata/output if available and then invoke `end()` exactly once without kwargs

### Requirement: No double-end behavior
The system SHALL prevent double-ending observations in the success path and SHOULD avoid telemetry warnings for already-ended spans.

#### Scenario: Single end invocation
- **WHEN** an observation completes normally
- **THEN** the helper SHALL call `end()` at most once and SHALL NOT trigger a secondary end that yields warnings
