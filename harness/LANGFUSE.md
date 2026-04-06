# Langfuse-First Usage

## Environment
- Required to enable: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
- Optional: `LANGFUSE_BASE_URL`, `LANGFUSE_TRACING_ENVIRONMENT`, `LANGFUSE_RELEASE`, `LANGFUSE_DEBUG=True` for verbose rollout, `LANGFUSE_ENABLED=0` to disable
- Model config remains: `CEREBRAS_API_KEY`, `RUNNER_MODEL`, `WRITER_MODEL`
- Span filter: keep Langfuse v4 default (no infra OTEL spans). Do not set `should_export_span` unless you need HTTP/DB/queue instrumentation.

## Benchmark Model (localization-native)
- Langfuse datasets remain the canonical benchmark source.
- Localization tasks use `LCATaskIdentity` (dataset/config/split + owner/repo + base_sha + issue/pull) and `changed_files` gold.
- Baselines/experiments operate on localization payloads (no symbols); metrics are file-set localization scores.

## Datasets
- `fetch_localization_dataset(name, version=None, dataset_config=None, dataset_split=None)` pulls hosted localization tasks.
- `fetch_localization_dataset_from_source(source, name, version=None, dataset_config=None, dataset_split=None)` supports Langfuse (default) or Hugging Face (`JetBrains-Research/lca-bug-localization`), using typed `LocalizationDatasetSource`. HF loader supports config/split/revision and surfaces failure categories (auth/missing/schema/load).
- `local_dataset(name, items, ...)` converts local task dicts to `LCATask` instances (normalizes identity/context/gold).
- Items must include localization identity fields and changed_files; symbol fields are ignored.

## Baselines & Experiments (Localization-first)
- Baseline bundle: reusable localization snapshots keyed by localization identity; contains predictions, metrics, repo metadata, and trace/run ids.
- Hosted localization baselines/experiments use the documented SDK via `run_hosted_localization_baseline` / `run_hosted_localization_experiment`.
- Hosted runs accept dataset_source; Langfuse remains default; Hugging Face path uses HF loader and HF repo materializer, emitting dataset source and materialization events in telemetry and stdout summaries.
- Local runs use `run_localization_task` to materialize, predict, and score; hosted runs emit dataset run ids and telemetry.
- Cost/usage: Cerebras generations must include `model` and usage (OpenAI-style prompt/completion/total). If the provider omits usage, supply it explicitly; do not emit zeroed usage. Costs are inferred via Langfuse model definitions.

## Tracing & Scoring

Orchestration-scoped tracing is owned by the state machines and their listeners:
- `loop_machine.py`: `OptimizationStateMachine` and `RepairStateMachine` drive the optimization and repair loops.
- `loop_listeners.py`: `OptimizationTracingListener` owns iteration spans (open/close, metric recording). `RepairTracingListener` owns the full repair attempt span lifecycle — opens spans on transition into `generating_candidate` (via `on_transition`, which fires before the machine's `on_enter` callback) and closes them on `retrying` / `candidate_valid` / `repair_exhausted`. Iteration metrics now record only `metrics.score` (no legacy metrics).
- `loop.py`: thin bootstrap/wiring layer — constructs context, dependencies, and the optimization machine; converts iteration records to public output shape. Not responsible for tracing or orchestration.

Leaf-operation observability remains with:
- `runner.py`: model calls and MCP tool invocations traced via Langfuse spans.
- `writer.py`: policy generations/retries traced; emits `writer.policy_generated` and `writer.policy_compiled`.
- `pipeline` steps emit exit-code/pass flags per step.
- `scorer.py` remains the source of metric computation; Langfuse stores the metrics.

## CLI
- Localization single run: `python run.py --mode localization-single --repo-owner ... --repo-name ... --base-sha ... --issue-title ... --issue-body ... --changed-file path1 --changed-file path2`
- Localization baseline (hosted dataset): `python run.py --mode localization-baseline --dataset DATASET_NAME [--version v1] [--config py] [--split dev] [--baseline-path bundle.json]`
- Localization experiment (hosted dataset): `python run.py --mode localization-experiment --dataset DATASET_NAME [--version v1] [--config py] [--split dev]`
- Console output is concise; Langfuse stores detailed traces/scores. `flush_langfuse()` runs on exit.
- To sync Cerebras pricing to Langfuse Cloud Models API: `python -c "from harness.observability.cerebras_pricing import sync_cerebras_models; sync_cerebras_models()"` (requires Langfuse Cloud credentials). Pricing entries live in `observability/cerebras_pricing.py`.

## Policy Loading
- `policy_loader.py` now loads `policy.py` and requires the public Iterative Context scorer contract `score(node: GraphNode, graph: Graph, step: int) -> float` (SelectionCallable). No sandbox or restricted execution.
