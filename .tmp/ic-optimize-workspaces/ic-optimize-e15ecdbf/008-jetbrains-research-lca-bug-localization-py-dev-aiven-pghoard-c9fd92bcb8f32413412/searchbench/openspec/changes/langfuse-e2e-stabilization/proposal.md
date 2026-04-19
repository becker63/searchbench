## Why

Real Langfuse runs of the canonical HF baseline command are failing despite stubbed tests passing. Observation lifecycle quirks, score emission incompatibilities, oversized propagated metadata, and malformed outbound chat requests all surface only against the live Langfuse API and real credentials. We need the hosted baseline path to be stable end-to-end now that tracing is part of the supported surface.

## What Changes

- Harden Langfuse observation lifecycle handling so context-manager and plain objects update/end safely without double-ending or unsupported kwargs.
- Make telemetry and score emission best-effort: failures in Langfuse scoring must not abort localization execution.
- Compact propagated tracing metadata to avoid Langfuse drops while keeping rich details attached where appropriate.
- Add local validation for outbound model requests so empty/invalid message lists are caught before provider submission.
- Iterate against the canonical one-item HF baseline command until it completes successfully with real Langfuse enabled.

## Capabilities

### New Capabilities
- `langfuse-lifecycle-compatibility`: Centralized, compatible observation lifecycle handling (start, update, end) that works with real Langfuse objects and stubs.
- `telemetry-resilience`: Best-effort Langfuse score/telemetry emission that never blocks localization execution.
- `metadata-propagation-compaction`: Controlled propagation of tracing metadata to stay within Langfuse limits while preserving detailed context in appropriate places.
- `outbound-request-validation`: Deterministic validation of outbound chat/completions requests (e.g., non-empty messages) before hitting providers.

### Modified Capabilities
- `<none>`: No existing capability specs are being altered; these are additions.

## Impact

- Code: `harness/observability/langfuse.py`, `harness/observability/score_emitter.py`, `harness/observability/experiments.py`, `harness/observability/baselines.py`, `harness/localization/executor.py`, `harness/loop/*`, `writer.py`, and related tests under `harness/tests_harness/`.
- Behavior: Observation lifecycle, telemetry error handling, metadata propagation, outbound request validation.
- External: Interaction with the real Langfuse API using existing credentials; Hugging Face baseline CLI remains unchanged.
