# LCA Bug Localization (Localization-First Runner)

## Canonical Contract
- **Gold**: `changed_files` (excluding tests).
- **Prediction**: list/set of file paths only.
- **Scoring**: Pydantic score bundles from static-graph distance components plus best-effort token diagnostics. The default composed control score uses graph-distance components; token efficiency is visible/diagnostic unless explicitly composed.
- **Evidence**: optional diagnostics (spans/snippets), not part of canonical prediction or scoring inputs.

## Identity
- Deterministic ID: dataset name/config/split + repo_owner/name + `base_sha` + issue/pull URL key.

## Runtime Context
- Issue title/body, diff_url/diff, repo_language[s]/license/stars, head_sha (supplemental).
- Delivered localization-native; no candidate IDs.

## Execution Architecture (Two-Phase)
- **Phase A (exploration)**: backend MCP tools -> OpenAI tool-calls -> backend `dispatch(...)`; gathers observations grounded in repo/issue/diff.
- **Phase B (finalization)**: separate no-tools request with strict JSON schema (`predicted_files`, `reasoning`, optional `evidence`); enforces shape via `response_format: json_schema`.
- **Evaluation**: score structured prediction against gold `changed_files` through static graph resolution; telemetry emits score bundle results post-run.

## Repo Materialization
- Inputs: dataset/config/split, repo_owner/name, `base_sha`.
- HF source: cached bare git mirror + per-task worktree checkout at `base_sha` with per-key lock, completion marker, and cache validation (worktree HEAD is checked against `base_sha` on reuse); stale locks and incomplete caches are pruned under the lock.
- Outputs: local path at `base_sha`, cache hit/miss, cache_validated, mirror_fetched, worktree_created, checkout_verified, logs/events.
- CLI logs: dataset/config/split, dataset source (langfuse|huggingface), cache hit/miss, git fetch/worktree/verification status, checkout `base_sha`, ready path, summary printed to stdout.
- Telemetry: includes dataset source and materialization events; failure categories surfaced (lock, auth/fetch, checkout verification, cache corruption).

## Dataset Source (HF-only public CLI)
- Public CLI baseline/experiment commands use the Hugging Face LCA dataset (`JetBrains-Research/lca-bug-localization`) directly, resolved via config/split/revision.
- Repo materialization still locks/cache-validates HF mirrors/worktrees per task (repo + base_sha), and logs materialization events as before.

## CLI Examples (HF-only)
- Baseline: `python run.py baseline --config py --split dev --max-items 25`
- Experiment: `python run.py experiment --config java --split test --max-items 50`
- Projection only: `python run.py experiment --config py --split dev --max-items 25 --projection-only`
- Pinned revision: `python run.py baseline --config py --split dev --revision 4b7c1d2 --max-items 20`
- Non-interactive: `python run.py experiment --config py --split dev --max-items 10 --yes`

## Dataset Guardrails (Baselines/Experiments)
- **Deterministic selection:** Tasks are sorted by task_id (dataset/config/split + repo owner/name + base_sha + issue/pull) before applying `--offset` (default 0) and `--max-items`. Selection metadata (offset, limit, selected count, total, identity preview) is printed and propagated.
- **Limits:** `--max-items N` bounds the window; `--offset K` skips K items before the window. Negative/zero limits are rejected. If `--max-items` is omitted, the run is unbounded.
- **Warnings:** If selection is unbounded or resolves to >50 items, the CLI emits a warning before projection/confirmation.
- **Cost projection (always-on):** Before running, the CLI prints planned and hard-cap projections using fixed bounds: localization (llama3.1-8b) 75k/10k tokens per item (planned) and 110k/14k (hard cap); writer (gpt-oss-120b) 20k/4096 tokens per call with 9 calls/iteration and 5 iterations hard cap. Pricing uses Cerebras tables only; unknown pricing fails closed. No user-provided assumptions or price overrides are accepted.
- **Confirmation:** After projection, the CLI requires confirmation (spacebar semantics). On TTY it reads a single key; if raw capture is unavailable it falls back to a line prompt requiring an explicit token (not a bare Enter). Ctrl+C/EOF aborts. Non-interactive runs must set `--yes` **and** `--max-items`.
- **Projection-only:** `--projection-only` prints selection + projection, runs confirmation/bypass logic, and exits without executing tasks.
- **Parallel workers:** `--max-workers` (default 1) controls only the outer dataset task execution after selection/projection/confirmation. Effective workers are `min(max-workers, selected_count)`; ordering is restored to deterministic selection order. A missing parent trace in worker execution is treated as an error rather than creating a new root span.

## Shared Evaluation Backend (Machine + Hosted)
- A shared localization evaluation backend (`evaluate_localization_batch`) executes per-task localization via `run_localization_task`, aggregates scores, and surfaces typed failures (`materialization`, `runner`, `scoring`, `unknown`) without string matching.
- The optimization/repair state machine invokes the backend through the evaluation hook (`evaluate_policy_on_item`) under a machine-owned evaluation span; LCA backend opens child task/materialization spans.
- Hosted baseline/experiment wrappers reuse the same backend and remain imperative shells; they emit localization-native score bundle summaries and outputs.
- Machine-facing scores use explicit `machine_score` semantics: `machine_score == score_summary.aggregate_composed_score`. There is no policy selector over file-set fields.
- Rollback should be treated as an implementation revert/emergency only, not a co-equal path; the shared backend remains the canonical evaluation route.
- Post-hoc policy-selection reducers run outside the loop to compare candidate policies; they consume `TaskScoreSummary` plus token-usage metadata with explicit availability flags. Reducer outputs are parallel to localization control scores and do not alter loop control. Reducer quality guards enforce configured score-name floors, defaulting to the composed score.
