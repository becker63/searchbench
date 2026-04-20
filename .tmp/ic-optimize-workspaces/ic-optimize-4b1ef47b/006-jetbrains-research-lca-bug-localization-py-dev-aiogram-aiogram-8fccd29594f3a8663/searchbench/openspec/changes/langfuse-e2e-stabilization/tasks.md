## 1. Observation lifecycle compatibility

- [x] 1.1 Centralize observation start/end handling in `harness/observability/langfuse.py` to cover context managers and plain objects with single-end semantics.
- [x] 1.2 Refactor callers (loop listeners, runner/writer, baselines) to use the centralized shutdown helper and remove direct `.end(error=...)` assumptions.
- [x] 1.3 Add/extend tests for startup/shutdown compatibility and double-end avoidance.

## 2. Telemetry resilience

- [x] 2.1 Harden score emission in `harness/observability/score_emitter.py` to be best-effort and tolerate missing client capabilities.
- [x] 2.2 Update hot-path callers (localization executor, baselines, writer) to treat telemetry failures as non-fatal while recording context where possible.
- [x] 2.3 Add tests covering telemetry failure paths without aborting localization logic.

## 3. Metadata propagation compaction

- [x] 3.1 Implement compact propagation of selection/projection metadata with detailed data attached only to root observation metadata or logs.
- [x] 3.2 Add tests to confirm oversized metadata is summarized for propagation.

## 4. Outbound request validation & E2E gate

- [x] 4.1 Add local validation for outbound chat/completions requests (non-empty messages) with caller context in errors for runner/writer paths.
- [x] 4.2 Add tests for validation failures with clear path identifiers.
- [ ] 4.3 Run and iterate on `uv run python run.py baseline --config py --split dev --max-items 1 --yes` until green; document the smoke check as part of stabilization.
