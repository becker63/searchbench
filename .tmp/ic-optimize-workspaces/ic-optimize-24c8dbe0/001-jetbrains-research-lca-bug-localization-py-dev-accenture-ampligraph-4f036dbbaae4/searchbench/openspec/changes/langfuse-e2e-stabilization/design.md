## Context

Langfuse tracing is a supported surface for the HF baseline CLI, but live runs diverge from stubbed tests: observation shutdown signatures differ, telemetry failures propagate, propagated metadata is oversized, and malformed chat requests reach providers. The public CLI and localization backend stay as-is; we need a compatibility layer and guardrails that work with the real Langfuse SDK and existing credentials.

## Goals / Non-Goals

**Goals:**
- Centralize observation lifecycle handling (start/update/end) in `harness/observability/langfuse.py` so both context managers and plain objects behave correctly.
- Make score/telemetry emission best-effort: failures must not abort localization execution.
- Compact propagated metadata to remain within Langfuse limits while preserving diagnostics elsewhere.
- Validate outbound chat requests before provider submission with context-rich errors.
- Keep the canonical one-item baseline command green against real Langfuse.

**Non-Goals:**
- No CLI surface changes or new user-facing flags.
- No replacement of Langfuse or large observability redesign.
- No new mock layers; rely on real Langfuse plus existing stubs for fast tests.

## Decisions

- **Lifecycle helper consolidation:** Route all observation start/end paths through shared helpers in `langfuse.py`, covering context-manager vs plain objects and update-before-end semantics. Avoid per-caller shutdown logic.
  - Alternative: Fix individual call sites; rejected due to drift risk and duplicated compatibility logic.
- **Safe end without kwargs:** Introduce `_safe_end_observation` that records error/output via `update(...)` when available and invokes `end()` with no kwargs. Prevent double-end and unsupported kwarg errors.
  - Alternative: Broad try/except around `.end(error=...)`; rejected because it hides errors and keeps divergent behavior.
- **Telemetry as best-effort:** Wrap score emission in compatibility checks; failures are logged/attached to observations but never raise into business logic. Localization metrics remain authoritative outside Langfuse.
  - Alternative: Force strict telemetry success; rejected because tracing should not block localization.
- **Metadata compaction:** Limit propagated attributes to compact summaries (e.g., selection/projection digests) and attach full detail only to root metadata or stdout. Prevent Langfuse drops while retaining diagnostics.
  - Alternative: Keep full payloads and accept Langfuse drops; rejected because warnings hide real issues and lose correlation.
- **Outbound request validation:** Validate non-empty `messages` and basic shape before hitting OpenAI-compatible clients; errors include backend/caller identifiers for debugging.
  - Alternative: Rely on provider errors; rejected because they obscure root cause and waste retries.
- **E2E gate:** Treat the canonical one-item baseline command as the primary validation loop; iterate until green.

## Risks / Trade-offs

- **Risk:** Overly aggressive metadata compaction could remove useful context. → Mitigation: keep rich detail on root metadata/stdout while propagating only summaries.
- **Risk:** Best-effort telemetry might mask systemic Langfuse outages. → Mitigation: log/attach failure context to observations and consider a lightweight warning counter.
- **Risk:** Centralized helpers might miss niche call sites. → Mitigation: repo-wide search for `.end(` usages and update tests to cover plain vs context-manager handles.
- **Risk:** Request validation could block valid-but-rare payloads. → Mitigation: scope validation narrowly (non-empty messages) and include clear errors for quick adjustment.

## Migration Plan

- Implement helpers and refactors behind existing APIs; no config changes.
- Update tests to cover lifecycle/telemetry/validation seams.
- Iterate the canonical baseline command until green; no data migrations required.
- Rollback: revert code changes; no persistent migrations involved.

## Open Questions

- Should telemetry failures surface as structured warnings in CLI output beyond logs?
- Do we need additional provider-specific validation beyond empty `messages` (e.g., tool call shape) in this pass?
