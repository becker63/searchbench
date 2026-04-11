
• Langfuse Inventory (inter-proc taint, tests included)
  Source: .tmp/langfuse_inventory_full.json (query in .tmp/langfuse-pack/src/
  langfuse_inventory.ql)

  ———

  ### observability/langfuse.py

  - <module>: imports langfuse; names Langfuse, LangfuseOpenAI, propagate_attributes,
    get_langfuse_client, get_tracing_openai_client, propagate_context, start_root_observation,
    start_child_observation, start_observation, flush_langfuse, _safe_end_observation,
    _safe_end_span.
  - _build_client: Langfuse (L107).
  - is_langfuse_enabled: get_langfuse_client (L126).
  - get_tracing_openai_client: LangfuseOpenAI (L136).
  - propagate_context: propagate_attributes (L211).
  - _safe_end_observation: lifecycle-call update (L247).
  - start_root_observation: get_langfuse_client (L277).
  - __exit__: lifecycle-call end (L311).
  - start_child_observation: attr start_observation; flows → trace, obs, step_obs, tool_obs,
    backend_obs, writer_obs (L349).
  - start_observation: flows → trace, obs, step_obs, tool_obs, backend_obs, writer_obs (L370);
    additional flows at L380.
  - flush_langfuse: get_langfuse_client (L393).

  ### observability/init.py

  - <module>: flush_langfuse, get_langfuse_client, get_tracing_openai_client, propagate_context,
    start_child_observation, start_observation, start_root_observation, emit_score,
    emit_score_for_handle.

  ### observability/score_emitter.py

  - <module>: emit_score, emit_score_for_handle, get_langfuse_client (L43, L57, L61, L94, L109).
  - emit_score: get_langfuse_client (L57, L61).
  - emit_score_for_handle: emit_score (L109).

  ### observability/baselines.py

  - <module>: LocalizationEvaluationError (L16); start_observation (L27); emit_score_for_handle
    (L28); flow trace (L81).
  - compute_baseline_for_task: flow trace; start_observation; LocalizationEvaluationError;
    emit_score_for_handle (L81, L105, L126).

  ### observability/datasets.py

  - <module>: get_langfuse_client (L14, L62).
  - fetch_localization_dataset: get_langfuse_client (L62).

  ### observability/cerebras_pricing.py

  - <module> / sync_cerebras_models: get_langfuse_client (L105, L107).

  ### observability/experiments.py

  - <module>: LocalizationEvaluationError (L18); flush_langfuse (L21); propagate_context (L21,
    L126); flush_langfuse (L160).
  - _baseline_worker: LocalizationEvaluationError (L187).
  - _run_experiment_task: LocalizationEvaluationError (L285).
  - run_hosted_localization_baseline: propagate_context, flush_langfuse (L126, L160).
  - run_hosted_localization_experiment: propagate_context, flush_langfuse (L356, L381).

  ### observability/baselines.py (flows noted above)

  - Flow: trace from start_observation (L81).

  ### localization/executor.py

  - <module>: LocalizationEvaluationError (L7, L56, L62); start_observation (L27);
    emit_score_for_handle (L28).
  - _resolve_repo_path: LocalizationEvaluationError (L56, L62).
  - _run_runner: LocalizationEvaluationError (L82, L85, L89).
  - _score_task: LocalizationEvaluationError (L137, L140).
  - _emit_telemetry: emit_score_for_handle (L167).
  - run_localization_task: flow trace; start_observation; LocalizationEvaluationError (L213,
    L229).

  ### localization/evaluation_backend.py

  - <module> / evaluate_localization_batch / _derive_machine_score: LocalizationEvaluationError
    (multiple: L10, L98, L115, L145, L171).

  ### localization/errors.py

  - <module>: LocalizationEvaluationError (L13).

  ### loop/runner_agent.py

  - <module>: _safe_end_observation, get_tracing_openai_client, start_observation,
    emit_score_for_handle (L14–18); get_tracing_openai_client (L268).
  - _make_client: get_tracing_openai_client (L268).
  - _collect_tool_results: flow tool_obs; start_observation; lifecycle update (L332–343).
  - run_agent: flow obs; start_observation; lifecycle update (L408–488).
  - _finalize_localization: flow obs; start_observation; lifecycle update (L717–742).
  - run_ic_iteration: flow backend_obs; start_observation; _safe_end_observation;
    emit_score_for_handle; lifecycle update (L779–802).
  - run_jc_iteration: flow backend_obs; start_observation; _safe_end_observation;
    emit_score_for_handle; lifecycle update (L846–869).

  ### loop/loop_machine.py

  - <module>: _safe_end_observation (L23, L326, L383); attrs start_observation (L306, L365).
  - _ensure_run_trace: attr start_observation (L306).
  - _finalize_run_trace: _safe_end_observation (L326).
  - on_enter_evaluating: attr start_observation; _safe_end_observation (L365, L383, L386).

  ### loop/loop_listeners.py

  - <module>: _safe_end_observation (L13, L61, L128); emit_score_for_handle (L14); attrs
    start_observation (L35, L78).
  - on_transition: attr start_observation (L35).
  - _end_attempt: _safe_end_observation (L61).
  - on_enter_evaluating: attr start_observation (L78).
  - _finalize_iteration: _safe_end_observation (L128).
  - _record_iteration_scores: emit_score_for_handle (L144, L154).

  ### loop/loop.py

  - <module>: flush_langfuse, propagate_context, _safe_end_observation, start_observation,
    emit_score (L18–23); lifecycle update (_hash_tests_dir L166–167).
  - prepare_iteration_tasks: _safe_end_observation (L237, L240).
  - run_loop: propagate_context; _safe_end_observation (L614, L644).

  ### loop/loop_types.py

  - <module>: start_observation (L357).

  ### loop/init.py

  - <module>: emit_score (L19).

  ### pipeline/pipeline.py

  - <module>: flow step_obs; start_observation; emit_score_for_handle; lifecycle update (L11–53).
  - run: flow step_obs; start_observation; emit_score_for_handle; lifecycle update (L27–53).

  ### writer.py

  - <module>: _safe_end_observation, get_tracing_openai_client, start_observation,
    emit_score_for_handle (L14–15, L53).
  - _get_client: get_tracing_openai_client (L53).
  - generate_policy: flow writer_obs; start_observation; emit_score_for_handle;
    _safe_end_observation (L236–321).

  ### run.py

  - <module>: flush_langfuse (L38, L588).
  - main: flush_langfuse (L588).

  ### utils/comparison.py

  - <module> / build_comparison: lifecycle update (L27).

  ### Tests (tests_harness)

  - test_observability.py: attrs/lifecycle for get_langfuse_client, emit_score*,
    start_observation, _safe_end_observation, propagate_context, lifecycle update.
  - test_session_policy.py: start_observation, emit_score (session inheritance/conflict cases).
  - test_executor_helpers.py: LocalizationEvaluationError.
  - test_localization_evaluation_backend.py: LocalizationEvaluationError.
  - test_loop_state_machine_min.py: start_observation.
  - test_experiments_backend.py: LocalizationEvaluationError.
  - test_tracing_wrappers.py: start_observation.
  - test_writer.py: start_observation.

  Flow summary (inter-proc, via TaintTracking::Global)

  - observability/langfuse.py: start_child_observation/start_observation → trace, obs, step_obs,
    tool_obs, backend_obs, writer_obs.
  - loop/runner_agent.py: tool_obs, obs, backend_obs across collection/run/IC/JC paths.
  - pipeline/pipeline.py: step_obs from run.
  - writer.py: writer_obs from generate_policy.
  - observability/baselines.py: trace from compute_baseline_for_task.
  - localization/executor.py: trace from run_localization_task.

  If you want the raw per-function lines/kinds/details, the full listing above is sourced
  from .tmp/langfuse_inventory_full.json; I can also export to a dedicated markdown file or add
  path-query output.
