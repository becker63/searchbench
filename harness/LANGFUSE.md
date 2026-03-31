# Langfuse-First Usage

## Environment
- Required to enable: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
- Optional: `LANGFUSE_BASE_URL`, `LANGFUSE_ENV`, `LANGFUSE_ENABLED=0` to disable
- Model config remains: `CEREBRAS_API_KEY`, `RUNNER_MODEL`, `WRITER_MODEL`

## Benchmark Model
- Langfuse datasets are the canonical benchmark source.
- JC baselines are computed as separate experiments over the dataset (cached snapshots).
- IC optimization runs consume those JC baseline snapshots and perform the inner loop per item.

## Datasets
- `fetch_dataset_items(name, version=None)` pulls hosted datasets.
- `local_dataset(name, items)` converts local task dicts to Langfuse-style items (normalized ids).
- Items must include `repo` and `symbol`; ids are stable (dataset id if present, otherwise hash).

## Baselines & Experiments (Langfuse-first)
- Baseline bundle: reusable JC snapshots keyed by dataset name/version and item id; contains `jc_result`, `jc_metrics`, trace/run ids, and pointers to the hosted baseline run (`baseline_run_id`, optional run name).
- Hosted JC baselines use the documented SDK:
  - `dataset = get_langfuse_client().get_dataset(name, version=...)`
  - `dataset.run_experiment(name="jc_baseline", task=...)`
  - `run_hosted_jc_baseline_experiment(...) -> BaselineBundle` caches the hosted outputs.
- Hosted IC optimization wraps `run_loop` inside `dataset.run_experiment(...)`:
  - `run_hosted_ic_optimization_experiment(name, version=None, iterations=..., baselines=bundle, recompute_baselines=False)`
  - Baselines are NOT recomputed implicitly. Provide a bundle or set `recompute_baselines=True` (which re-runs the hosted JC baseline experiment).
- Local fallback paths use `client.run_experiment(data=..., task=...)`:
  - `run_local_jc_baseline_experiment(dataset)`
  - `run_local_ic_optimization_experiment(dataset, iterations=..., baselines=bundle, recompute_baselines=False)`
- Hosted runs create dataset runs/compare UX; local runs create traces/scores only. Per-item traces include dataset info, item id, baseline run id, and run kind (`jc_baseline` vs `ic_optimization`).

## Tracing & Scoring

Orchestration-scoped tracing is owned by the state machines and their listeners:
- `loop_machine.py`: `OptimizationStateMachine` and `RepairStateMachine` drive the optimization and repair loops.
- `loop_listeners.py`: `OptimizationTracingListener` owns iteration spans (open/close, metric recording). `RepairTracingListener` owns the full repair attempt span lifecycle — opens spans on transition into `generating_candidate` (via `on_transition`, which fires before the machine's `on_enter` callback) and closes them on `retrying` / `candidate_valid` / `repair_exhausted`. Scores like `metrics.score`, `metrics.coverage_delta`, `metrics.tool_error_rate`, and `pipeline_passed` are recorded by the listeners.
- `loop.py`: thin bootstrap/wiring layer — constructs context, dependencies, and the optimization machine; converts iteration records to public output shape. Not responsible for tracing or orchestration.

Leaf-operation observability remains with:
- `runner.py`: model calls and MCP tool invocations traced via Langfuse spans.
- `writer.py`: policy generations/retries traced; emits `writer.policy_generated` and `writer.policy_compiled`.
- `pipeline` steps emit exit-code/pass flags per step.
- `scorer.py` remains the source of metric computation; Langfuse stores the metrics.

## CLI
- Ad hoc IC loop: `python run.py --iterations 3`
- JC baseline experiment (hosted runner): `python run.py --jc-baseline --dataset DATASET_NAME [--version v1] [--baseline-path bundle.json]`
- IC optimization experiment (hosted runner): `python run.py --ic-optimize --dataset DATASET_NAME [--version v1] --iterations 5 --baseline-path bundle.json [--recompute-baselines]`
- Console output is concise; Langfuse stores detailed traces/scores. `flush_langfuse()` runs on exit.

## Policy Loading
- `policy_loader.py` now simply loads `policy.py`, validates a callable `score`, and adapts 1/2/3-arg callables to `score(node, state, context) -> float`. No sandbox or restricted execution.
