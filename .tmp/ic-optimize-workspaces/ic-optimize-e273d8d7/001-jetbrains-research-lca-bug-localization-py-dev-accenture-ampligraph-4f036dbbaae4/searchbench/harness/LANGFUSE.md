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

Langfuse Python v4 is observation-centric in this codebase:
- scores are first-class evaluation objects attached to observations
- observation metadata stores compact structured execution detail
- propagated attributes are restricted to documented correlation fields

Canonical evaluation observations:
- `localization_task` is the localization evaluation surface. It emits numeric canonical component scores when available, always emits canonical `<component>_available` boolean scores, and stores compact canonical `score_components` metadata with `available`, `value`, and `reason`.
- `optimize_iteration` is the optimizer evaluation surface. It emits `metrics.score`, `pipeline_passed`, canonical component availability scores, and every numeric/boolean additional metric in scope, plus compact iteration score-context metadata and canonical `score_components` metadata.

Backend spans remain diagnostic-only:
- `iterative_context` and `jcodemunch` record backend diagnostics such as node counts and task summaries.
- They do not duplicate localization hop, token, or component evaluation scores.

Orchestration-scoped tracing is owned by the state machines and listeners:
- `orchestration/machine.py`: optimizer and repair control flow.
- `orchestration/dependencies.py`: concrete collaborator wiring for tracing-aware dependencies.
- `orchestration/evaluation.py`: policy-on-item evaluation against the shared localization runtime.
- `orchestration/writer.py`: repair writer support and policy file writes.
- `orchestration/listeners.py`: repair attempt span lifecycle.

Metadata channel rules:
- Use `propagate_context()` only for correlation context such as `trace_name`, `session_id`, `user_id`, `tags`, and similarly small documented attributes.
- Put execution diagnostics, score context, reducer summaries, repo paths, and task summaries on observation-local metadata.
- Observation metadata may be structured JSON. Propagated metadata must respect Langfuse's propagation limits.

## CLI
- Localization baseline (hosted Langfuse dataset): `python run.py baseline --dataset DATASET_NAME [--version v1] --max-items 25`
- Localization experiment (hosted Langfuse dataset): `python run.py experiment --dataset DATASET_NAME [--version v1] --max-items 25`
- Hugging Face ingestion: `python run.py sync-hf --dataset DATASET_NAME --config py --split dev [--revision REV]`
- Console output is concise; Langfuse stores detailed traces/scores. `flush_langfuse()` runs on exit.
- To sync Cerebras pricing to Langfuse Cloud Models API: `python -c "from harness.telemetry.tracing.cerebras_pricing import sync_cerebras_models; sync_cerebras_models()"` (requires Langfuse Cloud credentials). Pricing entries live in `telemetry/tracing/cerebras_pricing.py`.

## Policy Loading
- `policy_loader.py` now loads `policy.py` and requires the public Iterative Context frontier priority contract `frontier_priority(node: GraphNode, graph: Graph, step: int) -> float` (SelectionCallable). No sandbox or restricted execution.
