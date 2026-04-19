## Why

Harness localization scoring today only evaluates set overlap between predicted files and gold changed files. It cannot tell whether predictions are close to the issue anchors or reward solutions that stay concise on token usage. A small, deterministic hop-distance signal is needed so evaluation can reflect “how close” the prediction is within the repo topology and balance that against past token spend.

## What Changes

- Add a harness-local, read-only static graph built from the materialized repo snapshot via a small harness wrapper over `llm-tldr` (Tree-sitter-derived), producing language-agnostic nodes/edges for files, symbols/functions, imports, calls, contains/declared-in to support deterministic hop-distance queries.
- Provide deterministic anchor extraction order (backticked/explicit paths, slashy path-like tokens, symbol-like identifiers, code-block identifiers) and map anchors plus predicted files into the static graph for shortest-path hop calculations, marking unresolved items explicitly.
- Project hop distance and previous-round token usage into a separate projected scalar (`hop_token_score` or similar) while keeping legacy localization metrics (precision/recall/F1/hit) unchanged; allow the projection to be consumed for optimization without silently overwriting existing semantics.
- Reuse only the minimum iterative-context pieces as references (raw tree-ish schema, normalization to a `networkx.DiGraph`, read-only store/indexing for neighbor and hop queries); avoid traversal/runtime features.

## Capabilities

### New Capabilities
- `static-hop-eval-graph`: Build a harness-local static graph kernel from repo snapshots via a minimal `llm-tldr` wrapper, normalize nodes/edges, and expose deterministic hop-distance lookup from anchors to files.
- `localization-hop-score-projection`: Integrate hop-distance and prior token usage into localization scoring as a separate projected scalar alongside existing diagnostics.

### Modified Capabilities
- None.

## Impact

- Add a small static-graph kernel under `harness/localization` (new module/namespace) with a harness-local `llm-tldr` wrapper for ingest, normalization, indexing, and hop queries.
- Extend localization scoring and runtime plumbing (`harness/localization/scoring.py`, `harness/localization/runtime/execute.py`, `harness/localization/models.py`, possibly `harness/localization/runtime/records.py`) to incorporate hop-distance + token-usage projection while retaining current metrics and exposing the new scalar explicitly.
- Depend on existing lightweight graph utilities (e.g., `networkx` already in the repo); no traversal/frontier/MCP behavior is ported.
