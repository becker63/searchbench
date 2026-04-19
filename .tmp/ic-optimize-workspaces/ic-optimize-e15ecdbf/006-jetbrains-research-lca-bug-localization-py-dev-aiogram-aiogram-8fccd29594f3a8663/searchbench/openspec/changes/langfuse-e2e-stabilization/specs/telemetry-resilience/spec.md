## ADDED Requirements

### Requirement: Best-effort score emission
The system SHALL attempt to emit scores/telemetry to Langfuse, but failures or missing client capabilities SHALL NOT abort localization execution or baseline runs.

#### Scenario: Score API unavailable
- **WHEN** the active Langfuse client lacks a score API or raises an error during score creation
- **THEN** the system SHALL log or record the failure and continue the localization flow without raising a fatal exception

### Requirement: Error context captured without blocking
The system SHOULD attach relevant error context to observations when score emission fails while still allowing the business logic to proceed.

#### Scenario: Telemetry failure with context
- **WHEN** score emission fails for a specific observation
- **THEN** the system SHALL record the failure context on the observation if possible and SHALL continue processing the task
