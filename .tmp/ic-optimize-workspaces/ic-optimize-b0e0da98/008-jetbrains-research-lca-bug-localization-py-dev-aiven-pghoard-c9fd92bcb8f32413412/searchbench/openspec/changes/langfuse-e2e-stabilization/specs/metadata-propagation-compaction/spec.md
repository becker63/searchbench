## ADDED Requirements

### Requirement: Compact propagated metadata
The system SHALL limit propagated Langfuse attributes to compact summaries, avoiding oversized payloads that Langfuse drops.

#### Scenario: Large selection/projection inputs
- **WHEN** selection or projection data exceeds Langfuse attribute limits
- **THEN** the system SHALL propagate only compact summaries and SHALL keep detailed data attached to root observation metadata or logs instead of propagated attributes

### Requirement: Preserve diagnostic detail
The system SHOULD retain full diagnostic details outside propagation (e.g., root metadata or stdout) so troubleshooting remains possible without violating size limits.

#### Scenario: Detailed metadata retention
- **WHEN** large diagnostic metadata is present
- **THEN** the system SHALL store it in a non-propagated location while keeping propagated attributes within limits
