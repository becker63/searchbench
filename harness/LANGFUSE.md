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
- `fetch_localization_dataset(name, version=None)` pulls hosted localization tasks from Langfuse.
- Langfuse dataset items store the task payload in `input` and `changed_files` gold in `expected_output`.
- Hugging Face loading is sync-only ingestion tooling used to populate Langfuse datasets. Hosted runtime never loads tasks from Hugging Face.
- HF sync writes deterministic Langfuse item ids from repo owner/name, `base_sha`, and issue/pull identity; config, split, revision, and dataset name are provenance only.
- `local_dataset(name, items, ...)` converts local task dicts to `LCATask` instances (normalizes identity/context/gold).
- Items must include localization identity/context fields and `changed_files` gold; symbol fields are ignored.

## Baselines & Experiments (Localization-first)
- Evaluation bundles: derived localization run records keyed by task identity; contain predictions, score summaries, repo metadata, and trace/run ids. Canonical benchmark state remains in Langfuse datasets.
- Hosted localization baselines/experiments use the documented SDK via `run_hosted_localization_baseline` / `run_hosted_localization_experiment`.
- Hosted runs accept Langfuse dataset identity and optional version. `WorktreeManager` resolves repo@sha checkout caching independently from dataset identity.
- Local runs use `run_localization_task` to resolve a repo checkout, predict, and score; hosted runs emit dataset run ids and telemetry.
- Cost/usage: Cerebras generations must include `model` and usage (OpenAI-style prompt/completion/total). If the provider omits usage, supply it explicitly; do not emit zeroed usage. Costs are inferred via Langfuse model definitions.

## Tracing & Scoring

Orchestration-scoped tracing is owned by the state machines and their listeners:
- `orchestration/machine.py`: `OptimizationStateMachine` and `RepairStateMachine` drive the optimization and repair loops.
- `orchestration/dependencies.py`: typed dependency wiring for concrete collaborators used by the machines.
- `orchestration/evaluation.py`: policy-on-item evaluation against the shared localization backend.
- `orchestration/writer.py`: repair writer support, candidate cleanup, and policy file writes.
- `orchestration/listeners.py`: `OptimizationTracingListener` owns iteration spans (open/close, score recording). `RepairTracingListener` owns the full repair attempt span lifecycle — opens spans on transition into `generating_candidate` (via `on_transition`, which fires before the machine's `on_enter` callback) and closes them on `retrying` / `candidate_valid` / `repair_exhausted`. Iteration scoring records the composed score bundle output.

Leaf-operation observability remains with:
- `runner.py`: model calls and MCP tool invocations traced via Langfuse spans.
- `writer.py`: policy generations/retries traced; emits `writer.policy_generated` and `writer.policy_compiled`.
- `pipeline` steps emit exit-code/pass flags per step.
- `localization/scoring.py` remains the source of score-context construction; Langfuse stores composed score and named component results.

## CLI
- Localization baseline (hosted Langfuse dataset): `python run.py baseline --dataset DATASET_NAME [--version v1] --max-items 25`
- Localization experiment (hosted Langfuse dataset): `python run.py experiment --dataset DATASET_NAME [--version v1] --max-items 25`
- Hugging Face ingestion: `python run.py sync-hf --dataset DATASET_NAME --config py --split dev [--revision REV]`
- Console output is concise; Langfuse stores detailed traces/scores. `flush_langfuse()` runs on exit.
- To sync Cerebras pricing to Langfuse Cloud Models API: `python -c "from harness.telemetry.tracing.cerebras_pricing import sync_cerebras_models; sync_cerebras_models()"` (requires Langfuse Cloud credentials). Pricing entries live in `telemetry/tracing/cerebras_pricing.py`.

## Policy Loading
- `policy_loader.py` now loads `policy.py` and requires the public Iterative Context frontier priority contract `frontier_priority(node: GraphNode, graph: Graph, step: int) -> float` (SelectionCallable). No sandbox or restricted execution.
