## ADDED Requirements

### Requirement: Validate outbound chat requests
The system SHALL validate outbound chat/completions requests to ensure `messages` is a non-empty list before dispatching to providers.

#### Scenario: Empty messages list
- **WHEN** a localization runner or writer attempts to send a request with an empty `messages` list
- **THEN** the system SHALL raise a clear local error identifying the calling path and SHALL NOT send the request to the provider

### Requirement: Context-rich validation errors
Validation failures MUST include sufficient context (calling path or backend name) to diagnose which code path produced the malformed request.

#### Scenario: Identifying failing path
- **WHEN** validation rejects a malformed request
- **THEN** the error message SHALL include the relevant backend or caller identifier to aid debugging
