# LCA Bug Localization (Localization-First Runner)

## Canonical Contract
- **Gold**: `changed_files` (excluding tests).
- **Prediction**: list/set of file paths only.
- **Scoring**: file-set metrics (precision/recall/F1/hit); `metrics["score"]` maps to F1.
- **Evidence**: optional diagnostics (spans/snippets), not part of canonical prediction or scoring inputs.

## Identity
- Deterministic ID: dataset name/config/split + repo_owner/name + `base_sha` + issue/pull URL key.

## Runtime Context
- Issue title/body, diff_url/diff, repo_language[s]/license/stars, head_sha (supplemental).
- Delivered localization-native; no candidate IDs.

## Execution Architecture (Two-Phase)
- **Phase A (exploration)**: backend MCP tools -> OpenAI tool-calls -> backend `dispatch(...)`; gathers observations grounded in repo/issue/diff.
- **Phase B (finalization)**: separate no-tools request with strict JSON schema (`predicted_files`, `reasoning`, optional `evidence`); enforces shape via `response_format: json_schema`.
- **Evaluation**: score structured prediction against gold `changed_files`; telemetry/metrics emitted post-run.

## Repo Materialization
- Inputs: dataset/config/split, repo_owner/name, `base_sha`.
- HF source: cached bare git mirror + per-task worktree checkout at `base_sha` with per-key lock, completion marker, and cache validation (worktree HEAD is checked against `base_sha` on reuse); stale locks and incomplete caches are pruned under the lock.
- Outputs: local path at `base_sha`, cache hit/miss, cache_validated, mirror_fetched, worktree_created, checkout_verified, logs/events.
- CLI logs: dataset/config/split, dataset source (langfuse|huggingface), cache hit/miss, git fetch/worktree/verification status, checkout `base_sha`, ready path, summary printed to stdout.
- Telemetry: includes dataset source and materialization events; failure categories surfaced (lock, auth/fetch, checkout verification, cache corruption).

## Dataset Sources
- **Langfuse (default)**: hosted datasets fetched via `fetch_localization_dataset`.
- **Hugging Face**: `--dataset-source huggingface` (or request dataset_source=huggingface) loads `JetBrains-Research/lca-bug-localization` by config/split/revision, normalizes rows to `LCATask`, and materializes repos via git mirror/worktree. Networked tests are opt-in (`HF_NETWORK_TESTS=1`).
  - Recommended to pin dataset revision and surface it in logs; cache keys include dataset/config/split + repo + base_sha.

## CLI Examples
- Baseline (Langfuse): `python run.py --mode localization-baseline --dataset DATASET_NAME --config py --split dev`
- Baseline (HF): `python run.py --mode localization-baseline --dataset-source huggingface --dataset JetBrains-Research/lca-bug-localization --config py --split dev`
- Experiment (HF): `python run.py --mode localization-experiment --dataset-source huggingface --dataset JetBrains-Research/lca-bug-localization --config java --split test`
- Single (HF-backed requires repo materializer): `python run.py --mode localization-single --dataset-source huggingface --dataset JetBrains-Research/lca-bug-localization --repo-owner ... --repo-name ... --base-sha ...`
