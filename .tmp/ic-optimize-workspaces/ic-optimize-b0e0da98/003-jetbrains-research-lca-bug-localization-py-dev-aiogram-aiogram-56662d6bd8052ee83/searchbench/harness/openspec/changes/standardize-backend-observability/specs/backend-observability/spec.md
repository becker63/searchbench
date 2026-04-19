## ADDED Requirements

### Requirement: Backend-agnostic task summaries
The system SHALL produce a backend-agnostic task summary for every selected localization task attempted by IC or JC.

#### Scenario: Successful IC task summary
- **WHEN** an IC localization task completes successfully
- **THEN** the task summary includes backend `ic`, task identity, repo, status `success`, step count, tool call count, generation count, predicted file count, duration, and normalized token usage fields.

#### Scenario: Successful JC task summary
- **WHEN** a JC localization task completes successfully
- **THEN** the task summary includes backend `jc`, task identity, repo, status `success`, step count, tool call count, generation count, predicted file count, duration, and normalized token usage fields using the same field names as IC.

### Requirement: IC and JC task summary schema parity
The system SHALL use the same task summary schema, field names, field types, and field meanings for IC and JC.

#### Scenario: Successful backend summaries are schema-identical
- **WHEN** representative IC and JC tasks complete successfully
- **THEN** their serialized task summaries have the same required field names and each shared field has the same meaning for both backends.

#### Scenario: Failed backend summaries are schema-identical
- **WHEN** representative IC and JC tasks fail
- **THEN** their serialized failure summaries have the same required field names and each shared field has the same meaning for both backends.

#### Scenario: Backend-specific data exists
- **WHEN** a backend has extra diagnostic details that do not apply to the other backend
- **THEN** those details are kept outside the required task summary schema or placed in an explicitly optional metadata field without changing required schema parity.

### Requirement: Normalized token accounting
The system SHALL normalize provider usage details into fresh input, cached input, total input, output, and total token fields for every generation and task summary where usage is available.

#### Scenario: Cached token usage is present
- **WHEN** a generation usage record contains `usageDetails.input`, `usageDetails.input_cached_tokens`, `usageDetails.output`, and `usageDetails.total`
- **THEN** the normalized token breakdown reports `fresh_input_tokens` from `input`, `cached_input_tokens` from `input_cached_tokens`, `total_input_tokens` as their sum, `output_tokens` from `output`, and `total_tokens` from `total`.

#### Scenario: Cached token usage is absent
- **WHEN** a generation usage record contains fresh input and output tokens but no cached token field
- **THEN** the normalized token breakdown reports `cached_input_tokens` as zero and computes `total_input_tokens` from fresh plus cached input.

### Requirement: Step and generation counting semantics
The system SHALL define task step count as agent loop iterations and generation count as model generation attempts.

#### Scenario: Agent step is attempted
- **WHEN** IC or JC enters one `agent_step` iteration in the `run_agent(...)` loop
- **THEN** `step_count` increases by one for that task.

#### Scenario: Generation attempt occurs
- **WHEN** IC or JC attempts a model generation during an agent step, finalization, or retry
- **THEN** `generation_count` increases by one for that task.

#### Scenario: Model retry occurs
- **WHEN** retry logic performs another model request for the same logical agent step
- **THEN** `generation_count` increases for the retry while `step_count` does not increase unless another `agent_step` iteration begins.

### Requirement: Tool-call counting semantics
The system SHALL count all attempted backend tool invocations and partition them into successful and failed tool calls.

#### Scenario: Tool call succeeds
- **WHEN** IC or JC attempts dispatch of a named backend tool and receives a usable result
- **THEN** `tool_calls_count` and `successful_tool_calls_count` each increase by one.

#### Scenario: Tool call fails
- **WHEN** IC or JC attempts dispatch of a named backend tool and dispatch raises, returns a structured error, or otherwise fails
- **THEN** `tool_calls_count` and `failed_tool_calls_count` each increase by one.

#### Scenario: Tool call is retried
- **WHEN** IC or JC retries a backend tool invocation
- **THEN** each retry is counted as an additional attempted tool call.

#### Scenario: Tool call counts are reported
- **WHEN** a task summary is emitted
- **THEN** `tool_calls_count` equals `successful_tool_calls_count + failed_tool_calls_count`.

### Requirement: Standard execution phases
The system SHALL represent the same execution phases for IC and JC task traces: `backend_init`, `agent_execution`, `tool_execution`, and `finalization`.

#### Scenario: Backend initialization starts
- **WHEN** IC or JC begins constructing or resolving its backend runtime for a task
- **THEN** trace metadata identifies the phase as `backend_init` and includes backend name, task identity, and repo.

#### Scenario: Agent execution starts
- **WHEN** IC or JC begins model-driven agent execution for a task
- **THEN** trace metadata identifies the phase as `agent_execution` and includes backend name, task identity, and repo.

#### Scenario: Tool execution occurs
- **WHEN** IC or JC dispatches a backend tool call
- **THEN** the tool observation uses a `toolcall.<name>` span name and includes backend name, tool name, task identity, and repo when available.

#### Scenario: Finalization occurs
- **WHEN** IC or JC finalizes or recovers a localization prediction
- **THEN** trace metadata identifies the phase as `finalization` and includes backend name, task identity, and repo.

### Requirement: Failure-stage classification
The system SHALL classify localization task failures by the earliest known stage and expose that stage in logs, task summaries, and trace metadata.

#### Scenario: Backend initialization fails
- **WHEN** backend construction, tool listing, repository resolution, or repository indexing fails before the agent loop starts
- **THEN** the failure summary and trace metadata include status `failure`, stage `backend_init`, backend name, task identity, repo, and error message.

#### Scenario: Agent execution fails
- **WHEN** model invocation or agent-loop handling fails after backend initialization and before finalization
- **THEN** the failure summary and trace metadata include status `failure`, stage `agent_execution`, backend name, task identity, repo, and error message.

#### Scenario: Tool execution fails
- **WHEN** a backend tool dispatch fails during an agent step
- **THEN** the failure summary and trace metadata include status `failure`, stage `tool_execution`, backend name, task identity, repo, tool name, and error message.

#### Scenario: Finalization fails
- **WHEN** prediction finalization or fallback recovery fails after agent execution
- **THEN** the failure summary and trace metadata include status `failure`, stage `finalization`, backend name, task identity, repo, and error message.

### Requirement: Consistent CLI task output
The system SHALL render IC and JC task summaries through the same CLI formatter.

#### Scenario: Successful task CLI block
- **WHEN** an IC or JC task completes successfully during a run command
- **THEN** the CLI output includes a task block with fields in this order: backend and identity header, status, repo, steps/tool-calls/successful-tool-calls/failed-tool-calls/generations, normalized token fields, result predicted file count, and optional evaluation fields when available.

#### Scenario: Failed task CLI block
- **WHEN** an IC or JC task fails during a run command
- **THEN** the CLI output includes a task block with fields in this order: backend and identity header, status `failure`, failure stage, repo, steps/tool-calls/successful-tool-calls/failed-tool-calls/generations when known, normalized token fields when known, and error message.

#### Scenario: CLI fields are unavailable
- **WHEN** a required count or token field cannot be known for a task
- **THEN** the CLI formatter uses the same stable placeholder for IC and JC rather than omitting the field for one backend.

### Requirement: Optional evaluation metrics
The system MAY attach evaluation-derived metrics to task summaries when those metrics are already available, but those metrics MUST NOT be required for summary emission.

#### Scenario: Evaluation metrics are available
- **WHEN** scoring or evaluation has produced file-level metrics for a task
- **THEN** the task summary may include optional fields such as `gold_overlap_count`, file-level precision-style values, or file-level recall-style values.

#### Scenario: Evaluation metrics are unavailable
- **WHEN** a task fails before prediction or scoring, or evaluation metrics are otherwise unavailable
- **THEN** the task summary is still emitted with required observability fields.

### Requirement: Langfuse task summary metadata
The system SHALL attach the task summary and normalized token usage to Langfuse task-level telemetry for IC and JC runs.

#### Scenario: Successful task trace summary
- **WHEN** an IC or JC task completes successfully
- **THEN** the task-level Langfuse metadata includes backend, identity, repo, status, step count, tool call count, generation count, predicted file count, duration, and normalized token usage.

#### Scenario: Failed task trace summary
- **WHEN** an IC or JC task fails before producing observations
- **THEN** the task-level Langfuse metadata still includes backend, identity, repo, status `failure`, failure stage, and error message.

### Requirement: Run-level aggregate observability
The system SHALL aggregate `TaskRunSummary` records into a `RunSummary` for IC and JC run commands.

#### Scenario: Run completes with successful tasks
- **WHEN** an IC or JC run command completes one or more successful tasks
- **THEN** run-level output and metadata include backend name, selected task count, completed task count, successful task count, failed task count, success rate, aggregate normalized token usage, total tool calls, average tool calls per task, total generations, average generations per task, total duration, and average task duration.

#### Scenario: Run completes with failures
- **WHEN** an IC or JC run command completes with one or more failed tasks
- **THEN** run-level output and metadata include failed task count, failure stages, aggregate normalized token usage from completed generations, tool-call counts known before failure, duration known before failure, and backend name.

#### Scenario: Run summary source of truth
- **WHEN** a `RunSummary` is computed
- **THEN** all run-level totals and averages are derived from emitted `TaskRunSummary` records rather than from raw spans, raw observations, or provider aggregate usage.

#### Scenario: Run token averages are computed
- **WHEN** a `RunSummary` is computed from task summaries
- **THEN** average fresh input, cached input, total input, output, and total tokens per task are computed from the corresponding task summary totals.
