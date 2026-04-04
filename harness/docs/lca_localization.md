# LCA Bug Localization (Phase 1 Prep)

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

## Repo Materialization (Phase 2)
- Inputs: dataset/config/split, repo_owner/name, `base_sha`, archive refs.
- Outputs: local path at `base_sha`, cache/download/extract/checkout status/log hooks.
- CLI logs: dataset/config/split, repo resolve, cache hit/miss, download/fetch, extract/materialize, checkout `base_sha`, ready path.

## Phase 2 Boundaries
- No dataset loading, repo fetch/extract/checkout, or execution in phase 1.
- Phase 2 will implement materializer, dataset ingestion, and actual evaluation runs.
