## Context

Searchbench currently runs localization through IC and JC backends that share the same high-level evaluation path but expose different operational signals. Both backends eventually produce localization predictions and scoring artifacts, but their traces, log lines, token usage records, and failure metadata are not a stable contract. This makes backend comparisons hard to trust and makes early failures harder to debug.

Recent work moved `ic run` and `jc run` onto a shared canonical task reconstruction and materialization path, and JC backend initialization now emits some early failure metadata. This change extends that direction into a formal backend observability contract that applies to all localization backends.

The relevant runtime boundaries are:

- `agents/localizer.py` owns IC/JC backend orchestration, agent execution, finalization, and Langfuse child observations.
- `agents/common.py` already contains shared serialization and provider usage normalization helpers.
- `localization/runtime/evaluate.py` owns task execution and batch result aggregation.
- `entrypoints/cli.py` owns operator-facing run output.
- Langfuse tracing helpers in `telemetry/tracing` provide the existing observation/span abstraction.

## Goals / Non-Goals

**Goals:**

- Define one task-level summary shape for IC and JC localization runs.
- Define one run-level summary shape derived from task summaries.
- Normalize fresh, cached, output, and total token usage in one place.
- Define exact step and tool-call counting semantics.
- Emit comparable CLI task summaries for successful and failed IC/JC tasks.
- Attach the same summary metadata to Langfuse task/backend spans.
- Represent the same execution phases for both backends: `backend_init`, `agent_execution`, `tool_execution`, and `finalization`.
- Preserve early failure visibility when backend initialization fails before the agent loop starts.
- Preserve room for optional evaluation-derived summary metrics without making them required for every task.
- Enforce schema parity between IC and JC summaries through tests.
- Keep the change additive and localized to observability and result metadata.

**Non-Goals:**

- Do not change IC or JC backend algorithms.
- Do not change tool semantics or MCP server behavior.
- Do not modify Langfuse SDK behavior.
- Do not introduce new external dependencies.
- Do not define dashboards or long-term aggregate reporting storage.
- Do not make token accounting responsible for billing reconciliation beyond the normalized usage fields observed in provider responses.
- Do not require optional evaluation-derived metrics to exist for every run.

## Decisions

### Introduce backend-agnostic observability models

Add a small typed module such as `telemetry/observability_models.py` with:

- `TokenUsageBreakdown`
- `BackendRunStats`
- `TaskRunSummary`
- `RunSummary`

The models should use existing Pydantic patterns in the repository. They should be serializable to plain JSON metadata for logs, traces, and tests.

Alternative considered: add loose dictionaries directly in `agents/localizer.py`. That is rejected because the whole point of the change is a shared contract; untyped dictionaries would make drift between IC and JC easy.

### Enforce IC/JC schema parity

`TaskRunSummary` is the schema boundary for backend comparison. IC and JC must populate the same model. Backend-specific fields should be avoided in the required schema; if a field is present for one backend, it must have the same name, type, and meaning for the other backend. Backend-specific details can remain in raw observations or backend-specific metadata outside the required summary contract.

Parity tests should compare the dumped summary key set for representative IC and JC success and failure cases. They should not merely assert that both backends emit some summary.

Alternative considered: allow each backend to expose its own summary and normalize later. That is rejected because the comparison layer would become a second source of truth and would hide drift until analysis time.

### Normalize token usage centrally

Extend shared usage handling in `agents/common.py` or a narrow helper imported by it. The helper should accept provider/Langfuse-style usage details and produce:

- `fresh_input_tokens`
- `cached_input_tokens`
- `total_input_tokens`
- `output_tokens`
- `total_tokens`

The canonical mapping is:

- `fresh_input_tokens = usageDetails.input`
- `cached_input_tokens = usageDetails.input_cached_tokens`
- `total_input_tokens = fresh_input_tokens + cached_input_tokens`
- `output_tokens = usageDetails.output`
- `total_tokens = usageDetails.total`

If `usageDetails.total` is absent, compute it as `total_input_tokens + output_tokens`. If cached input is absent, treat it as zero. Preserve raw usage details separately when existing telemetry already stores them.

Alternative considered: infer cached tokens by subtracting known prompt lengths. That is rejected because provider usage details are the source of truth and inference would create fragile backend-specific behavior.

### Define run-level aggregation from task summaries

`RunSummary` should be computed from `TaskRunSummary` records, not by re-reading spans or raw observations. This keeps the run-level layer deterministic and testable. The run summary should include:

- `backend`
- `selected_tasks_count`
- `completed_tasks_count`
- `successful_tasks_count`
- `failed_tasks_count`
- `success_rate`
- total fresh input, cached input, total input, output, and total tokens
- average per-task fresh input, cached input, total input, output, and total tokens
- total tool calls and average tool calls per task
- total generations and average generations per task
- total duration and average task duration

Use selected task count for selection semantics and completed task count for summaries actually produced. A task that fails still contributes a `TaskRunSummary` and therefore contributes to completed/failed counts and any tokens/calls/duration known before failure.

Alternative considered: compute run totals from Langfuse aggregate usage fields. That is rejected because aggregate usage can mix cached and fresh tokens and may not preserve the per-task semantics needed for backend comparisons.

### Define step and generation counting

`step_count` should mean agent loop iterations, using the real `run_agent(...)` shape in `agents/localizer.py` as the reference. One `agent_step` iteration counts as one step once the loop attempts model-driven work for that iteration.

`generation_count` should count model generation attempts. If a retry causes another model request, the retry counts as an additional generation. Generation count may be greater than step count when retry logic performs multiple generation attempts for one logical step.

Alternative considered: define step count as generation count. That is rejected because the agent loop and model retry behavior answer different questions and should be separately visible.

### Define tool-call counting

Tool counts should be based on attempted backend tool invocations. A tool call is counted when the agent/runtime attempts dispatch of a named backend tool, regardless of whether the call succeeds.

The summary should include:

- `tool_calls_count`: all attempted tool invocations
- `successful_tool_calls_count`: attempted tool invocations that returned a usable tool result
- `failed_tool_calls_count`: attempted tool invocations that raised, returned a structured error, or otherwise failed dispatch

Retries count as additional attempted tool calls. Successful and failed counts should partition total attempted calls, so `tool_calls_count = successful_tool_calls_count + failed_tool_calls_count`. Counts should be derived from the same observations/spans that produce `toolcall.<name>` telemetry, not from backend-specific ad hoc counters.

Alternative considered: count only successful tool calls. That is rejected because failed dispatches are operationally important and hiding them would make backend reliability comparisons misleading.

### Treat phases as explicit telemetry stages, not separate control flows

IC and JC should both annotate spans and summaries with the same phase names:

- `backend_init`
- `agent_execution`
- `tool_execution`
- `finalization`

Implementation can keep the current localizer flow. The phase contract can be represented through span names, metadata fields, and failure-stage values without splitting the runtime into a new state machine.

Alternative considered: redesign `run_ic_iteration` and `run_jc_iteration` around a common executor class. That is out of scope because it risks algorithmic churn and broader regressions.

### Emit one task summary after every task attempt

Each completed or failed task should produce a `TaskRunSummary` with status, backend, identity, repo, counts, duration, token usage, prediction count, and failure details when applicable. The summary should flow to:

- CLI output through a shared formatter.
- Langfuse metadata on the relevant task/backend span.
- Runtime/evaluation result metadata if needed to render after batch execution.

Alternative considered: only log summaries from inside `agents/localizer.py`. That would hide summaries from the batch-level CLI formatting path and make failures surfaced by `evaluate_localization_batch` less consistent.

### Keep optional evaluation metrics explicit

`TaskRunSummary` may carry optional evaluation-derived metrics when they are already available, such as:

- `gold_overlap_count`
- file-level precision-like values
- file-level recall-like values
- other score-derived scalar fields that are already computed by the evaluation path

These fields must not block summary emission. A task summary is valid without evaluation metrics, especially for failures before prediction or scoring. Required summary fields remain focused on observability and comparability, not scoring redesign.

Alternative considered: require evaluation metrics on every task summary. That is rejected because early failures and pre-scoring failures still need summaries.

### Keep CLI output concise but structured

The CLI should continue printing aggregate run results, then one task-level block per selected item. Use the same formatter for IC and JC.

Task blocks should use stable ordering:

1. `[TASK] <backend> identity=<id>`
2. `status=<success|failure>`
3. `stage=<stage>` only for failures or when a non-success terminal stage is useful
4. `repo=<repo>`
5. `steps=<n> tool_calls=<n> successful_tool_calls=<n> failed_tool_calls=<n> generations=<n>`
6. `tokens: fresh_input=<n> cached_input=<n> total_input=<n> output=<n> total=<n>`
7. `result: predicted_files=<n>` for successes when prediction information exists
8. `evaluation: ...` only for optional evaluation metrics that are available
9. `message=<error>` for failures

Always show backend, identity, status, repo, count fields, and token fields. Use zero for known-zero counts and a stable placeholder such as `unknown` only when the value genuinely cannot be known. Optional evaluation fields are shown only when available. Failures should render the same field names as successes plus `stage` and `message`.

Alternative considered: emit JSON-only logs. That is rejected for now because current operator workflows rely on readable text output.

### Classify failures at the closest boundary

Failure stage should be set where the failure is first known:

- Backend constructor/list-tools/index failures: `backend_init`
- Model or agent loop failures: `agent_execution`
- Tool dispatch failures: `tool_execution`
- Final prediction construction failures: `finalization`

Evaluation-level wrapping should preserve this classification rather than replacing it with a generic runner failure when possible.

Alternative considered: classify failures after the fact from exception strings. That is rejected because it is brittle and loses precise context.

## Risks / Trade-offs

- [Risk] Some providers may omit cached token fields or use different key names. → Mitigation: normalize from known `usageDetails` keys and default missing cached tokens to zero while preserving raw usage data.
- [Risk] Adding summary fields can duplicate existing telemetry metadata. → Mitigation: keep summary fields stable and compact, and avoid copying full observations or prompts into summary metadata.
- [Risk] CLI output can become noisy for large runs. → Mitigation: print one concise task block per item and keep detailed per-step data in Langfuse.
- [Risk] Failure-stage classification may be incomplete at first. → Mitigation: start with the four required phases and default unknowns explicitly rather than silently omitting stage.
- [Risk] Existing tests may assert legacy output. → Mitigation: update tests to assert the shared formatter and keep aggregate summary lines stable where possible.

## Migration Plan

1. Add the shared observability models and token normalization helper.
2. Add task summary parity tests before wiring both backends, using the model as the schema boundary.
3. Update IC and JC localizer flows to populate phase metadata and construct summaries.
4. Define step, generation, and tool-call counting from agent loop observations/spans.
5. Thread task summaries through runtime/evaluation results where the CLI can render them after execution.
6. Build `RunSummary` aggregation from task summaries.
7. Replace backend-specific CLI task output with a shared formatter.
8. Add tests for IC/JC parity, token breakdown, count semantics, aggregation, failure-stage classification, and trace metadata.
9. Run focused backend, localization, CLI, tracing, and token usage tests.

Rollback is straightforward: remove the new summary propagation and formatter usage while retaining existing run behavior. The change should not alter task selection, backend algorithms, scoring, or materialization.
