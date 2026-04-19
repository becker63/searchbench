## ADDED Requirements

### Requirement: Compute hop-distance and token usage projection
The system SHALL compute `distance_term = 1 / (1 + hops)` for each prediction using the minimal hop distance (0 distance produces 1.0, unreachable produces 0.0) and `token_term = 1 / (1 + log1p(previous_total_tokens))` using prior token usage when available (otherwise treat as 1.0). The scorer SHALL combine these terms via a configurable weighted average to produce a projected scalar (e.g., `hop_token_score`) separate from legacy metrics.

#### Scenario: Composite scalar produced from distance and tokens
- **WHEN** hop distances and prior token usage are available for a localization evaluation
- **THEN** the scorer calculates distance_term and token_term using the defined formulas and emits a weighted composite scalar

### Requirement: Preserve legacy localization diagnostics
The system SHALL continue to compute and emit existing localization metrics (precision, recall, F1, hit) alongside the new composite scalar for diagnostics and comparison.

#### Scenario: Legacy metrics remain available
- **WHEN** localization scoring runs with the new projection enabled
- **THEN** precision, recall, F1, and hit are still calculated and surfaced with their original semantics in addition to the composite scalar

### Requirement: Projected scalar is emitted alongside legacy metrics
The system SHALL emit the projected hop/token scalar alongside legacy localization metrics without changing the semantics of precision, recall, F1, or hit; making the projection the primary optimization score MUST be gated/configurable.

#### Scenario: Projected score emitted without overwriting legacy metrics
- **WHEN** evaluation results are emitted for a task
- **THEN** the projected scalar is included for optimization/analysis, legacy metrics retain their existing meaning, and any switch to make the projected scalar primary is explicit/configurable
