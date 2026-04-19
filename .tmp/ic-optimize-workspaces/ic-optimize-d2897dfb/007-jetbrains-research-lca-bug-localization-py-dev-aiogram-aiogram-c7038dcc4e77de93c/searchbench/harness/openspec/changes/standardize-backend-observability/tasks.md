## 1. Shared Models And Token Normalization

- [x] 1.1 Add `telemetry/observability_models.py` with `TokenUsageBreakdown`, `BackendRunStats`, `TaskRunSummary`, and `RunSummary` Pydantic models.
- [x] 1.2 Add a central token usage normalization helper that maps provider `usageDetails` into fresh input, cached input, total input, output, and total token fields.
- [x] 1.3 Define `TaskRunSummary` required fields so IC and JC use identical names, types, and meanings.
- [x] 1.4 Define `RunSummary` fields for selected, completed, successful, failed, success rate, token totals, token averages, tool-call totals/averages, generation totals/averages, duration totals, and duration averages.
- [x] 1.5 Add unit tests for token normalization with cached-token fields present, cached-token fields absent, and total-token fallback computation.
- [x] 1.6 Add schema parity tests that compare serialized IC and JC summary key sets for success and failure cases.

## 2. Runtime Summary Construction

- [x] 2.1 Update IC execution to collect backend, identity, repo, status, step count, tool call count, generation count, predicted file count, duration, and normalized token usage into a `TaskRunSummary`.
- [x] 2.2 Update JC execution to collect the same `TaskRunSummary` fields as IC using identical field names and status values.
- [x] 2.3 Implement `step_count` as attempted `agent_step` loop iterations for both IC and JC.
- [x] 2.4 Implement `generation_count` as model generation attempts, with retries counted as additional generations.
- [x] 2.5 Implement `tool_calls_count`, `successful_tool_calls_count`, and `failed_tool_calls_count` from attempted tool dispatch observations/spans, with retries counted as additional attempts.
- [x] 2.6 Thread task summaries through `LocalizationRunResult` or the runtime evaluation result boundary so CLI and batch-level code can render them after task execution.
- [x] 2.7 Allow optional evaluation-derived fields such as gold overlap count and file-level precision/recall style values when available without blocking summary emission.

## 3. Phase And Failure Instrumentation

- [x] 3.1 Emit consistent phase metadata for `backend_init`, `agent_execution`, `tool_execution`, and `finalization` in IC traces.
- [x] 3.2 Emit consistent phase metadata for `backend_init`, `agent_execution`, `tool_execution`, and `finalization` in JC traces.
- [x] 3.3 Classify backend initialization failures as `backend_init` and include backend, task identity, repo, and error message in summary and trace metadata.
- [x] 3.4 Classify agent-loop, tool-dispatch, and finalization failures as `agent_execution`, `tool_execution`, and `finalization` respectively.
- [x] 3.5 Add tests that verify early backend-init failures produce structured metadata even when no agent loop runs.

## 4. Langfuse Metadata And CLI Output

- [x] 4.1 Attach successful task summaries and normalized token usage to task-level Langfuse metadata for IC and JC.
- [x] 4.2 Attach failed task summaries and failure-stage metadata to task-level Langfuse metadata for IC and JC.
- [x] 4.3 Add a shared CLI formatter for IC and JC task summaries with stable field ordering for header, status, stage, repo, counts, tokens, result, optional evaluation metrics, and failure message.
- [x] 4.4 Ensure the CLI formatter always shows required fields with the same labels for IC and JC, using a stable placeholder when a value is genuinely unknown.
- [x] 4.5 Update `ic run` and `jc run` output to render successful and failed task blocks through the shared formatter.
- [x] 4.6 Add run-level aggregate summary output and metadata for selected count, completed count, successful count, failed count, success rate, aggregate token usage, token averages, total/average tool calls, total/average generations, total duration, average task duration, and backend.

## 5. Run-Level Aggregation

- [x] 5.1 Implement `RunSummary` construction from emitted `TaskRunSummary` records as the only source of truth for run-level totals and averages.
- [x] 5.2 Aggregate token totals and per-task averages for fresh input, cached input, total input, output, and total tokens.
- [x] 5.3 Aggregate tool-call totals and averages, including attempted, successful, and failed tool-call counts.
- [x] 5.4 Aggregate generation totals and averages.
- [x] 5.5 Aggregate total duration and average task duration.
- [x] 5.6 Add tests proving aggregate totals and averages are derived from task summaries rather than raw spans or provider aggregate usage.

## 6. Tests And Verification

- [x] 6.1 Add CLI tests proving IC and JC task blocks use the same field names and stable ordering for success and failure.
- [x] 6.2 Add tracing tests proving required span names and metadata are emitted for both backends.
- [x] 6.3 Add runtime tests proving `step_count` follows agent loop iterations and `generation_count` follows model generation attempts including retries.
- [x] 6.4 Add runtime tests proving attempted, successful, and failed tool-call counts partition all attempted tool invocations.
- [x] 6.5 Add runtime tests proving run-level aggregate observability includes token totals, token averages, tool-call totals/averages, generation totals/averages, success count, failure count, success rate, total duration, and average task duration.
- [x] 6.6 Run focused tests for backend adapters, localization runtime, CLI guardrails, tracing wrappers, and token usage.
- [x] 6.7 Run lint/type checks for touched files.
