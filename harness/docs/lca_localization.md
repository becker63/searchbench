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
- Inputs: dataset/config/split, repo_owner/name, `base_sha`, archive refs.
- Outputs: local path at `base_sha`, cache/download/extract/checkout status/log hooks.
- CLI logs: dataset/config/split, repo resolve, cache hit/miss, download/fetch, extract/materialize, checkout `base_sha`, ready path.
