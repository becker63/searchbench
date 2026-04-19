## Why

IC and JC localization runs are comparable in purpose but not yet comparable in observability. Logs, trace structure, failure context, and token accounting differ enough that backend A/B analysis can be misleading, especially when cached prompt tokens are mixed with fresh prompt tokens in aggregate usage.

This change defines a shared observability contract so Searchbench can measure, debug, and optimize localization backends under the same task-level and run-level aggregation semantics.

## What Changes

- Add a backend-agnostic task observability model for localization runs.
- Add a backend-agnostic run summary model that aggregates task summaries into stable run-level totals and averages.
- Standardize task status, failure stage, backend name, identity, repo, step count, tool-call count, generation count, predicted-file count, duration, and token totals across IC and JC.
- Define counting semantics for agent steps, attempted tool calls, successful tool calls, failed tool calls, and retry behavior.
- Normalize token usage into fresh input, cached input, total input, output, and total tokens.
- Require IC and JC summaries to use schema-identical field names and semantics so backend comparisons are testable.
- Emit task summaries to both CLI logs and Langfuse task-span metadata.
- Emit run summaries to CLI logs and run-level metadata, derived from `TaskRunSummary` records.
- Align trace phases for IC and JC around backend initialization, agent execution, tool execution, and finalization.
- Classify early failures by stage so backend initialization failures remain visible even when no agent loop runs.
- Add focused tests for shared summary shape, schema parity, counting semantics, aggregation, token accounting, trace metadata, CLI output, and failure-stage classification.

## Capabilities

### New Capabilities

- `backend-observability`: Defines the shared task summary, run aggregation, token accounting, trace, logging, counting, schema parity, and failure-stage contract for localization backends.

### Modified Capabilities

None.

## Impact

- Affected code:
  - `agents/common.py` for token usage normalization helpers.
  - `agents/localizer.py` for IC/JC phase instrumentation and summary construction.
  - `entrypoints/cli.py` for consistent task summary rendering.
  - `localization/runtime/evaluate.py` and related runtime models for task summary propagation and run summary aggregation.
  - `telemetry/tracing` helpers if shared span metadata emission needs a small utility.
  - Tests under `tests_harness/`, especially backend adapter, localization, CLI guardrail, tracing, and token usage coverage.
- Affected systems:
  - Langfuse traces and metadata for localization runs.
  - Operator-facing CLI logs for `ic run` and `jc run`.
- Dependencies:
  - No new external dependencies.
