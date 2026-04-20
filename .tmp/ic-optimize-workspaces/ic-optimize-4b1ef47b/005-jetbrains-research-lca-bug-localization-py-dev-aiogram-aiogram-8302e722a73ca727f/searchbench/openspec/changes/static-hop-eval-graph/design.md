## Context

- Harness localization scoring currently operates on set overlap (precision/recall/F1/hit) between predicted files and gold `changed_files`. Token usage is captured but not factored into the primary score.
- Iterative Context provides a richer runtime graph, but only a small subset is relevant for evaluation: `raw_tree.py` (file/function/import/call structs), `normalize.py` (RawTree → `networkx.DiGraph`), `graph_models.py` (Graph alias; node/edge metadata), and `store.py` (read-only indexes plus neighbor traversal). The ingestion adapter (`injest/llm_tldr_adapter.py`) shows how call/import edges are populated; harness will build a smaller, evaluation-oriented wrapper over `llm-tldr` for static extraction only.
- Constraints: build a read-only static graph from the materialized repo snapshot; support deterministic shortest-hop queries from anchors to predicted files; do not copy traversal/frontier selection, MCP behavior, or runtime mutation semantics from Iterative Context. Keep the scoring surface pluggable but start with a simple scalar projection using hop distance and previous token usage.

## Goals / Non-Goals

**Goals:**
- Add a harness-local static graph kernel that constructs a normalized directed graph from the repo snapshot with file and symbol/function nodes plus import/call/declared-in edges.
- Provide deterministic anchor extraction and mapping of anchors/predictions into that graph, then compute minimal hop distances for evaluation.
- Integrate a projected scalar score combining hop distance and prior token usage alongside existing localization metrics (precision/recall/F1/hit) for diagnostics.
- Keep the kernel isolated under a new harness namespace to avoid coupling with existing runtime search logic.

**Non-Goals:**
- No traversal, frontier selection policy, expansion loops, MCP server integration, or active graph mutation semantics.
- No reimplementation of Iterative Context runtime events, score history, or traversal-time behaviors.
- No JC tool-result normalization or shadow graph; only deterministic static graph over the repo snapshot.
- No rich anchor NLP or multi-language semantic graphing in phase 1; keep anchors/edges simple and extendable later.

## Decisions

- **Module boundary:** Introduce `harness/localization/static_graph/` (or similarly named) with submodules for `ingest` (minimal harness-local `llm-tldr` wrapper), `raw_tree` (data models), `normalize` (RawTree → DiGraph), `store` (indexes + neighbor/shortest-path), `anchors` (extraction/mapping helpers), and optional `distance` utilities. Keep scoring integration changes in existing localization modules.
- **Graph model:** Reuse Iterative Context’s minimal pieces: node attributes (id, symbol, file, type/kind), edge kinds (`imports`, `calls`, `declared_in`/`contains`). Use `networkx.DiGraph` with attributes; no event subject or node state variants.
- **Ingestion scope:** Derive RawTree from the repo snapshot through a harness-local, evaluation-oriented wrapper over `llm-tldr` (Tree-sitter-derived), smaller than IC’s adapter:
  - File nodes for all source files under evaluation.
  - Function/symbol nodes and import/call/contains edges where `llm-tldr` provides them deterministically; avoid language-specific bespoke parsers and omit anything requiring LLMs or runtime execution.
  - Keep ingestion pluggable per language and degrade gracefully: when details are unavailable, retain file-level nodes and obvious import/include edges without fabricating data.
- **Normalization/indexing:** Normalize RawTree to DiGraph mirroring `normalize.raw_tree_to_graph`, then construct a read-only Store to support:
  - Index by symbol and file (forward/backward).
  - Neighbor lookup by edge kind.
  - Unweighted shortest-path/hop distance queries (BFS over Store edges) returning deterministic hop counts.
- **Anchor extraction/mapping:** Extract anchors deterministically, in order: (1) explicit/backticked file paths; (2) slash-containing path-like tokens; (3) exact/CamelCase/snake_case identifiers; (4) identifiers in code blocks; otherwise mark unresolved. Canonicalize paths via existing `canonicalize_path`. Resolve anchors to graph nodes via Store index; fall back to file-level anchors when symbols are unknown, and leave unresolved anchors marked as such.
- **Prediction mapping:** Map predicted files to graph nodes via canonicalized file paths; if a prediction does not resolve, mark it unresolved and treat its hop distance as unreachable rather than fabricating placeholder nodes.
- **Hop-distance computation:** For each predicted file node, compute the minimal hop distance from any resolved anchor using Store shortest-path; treat unreachable as `None`/`inf`. Expose both raw distance and normalized `distance_term = 1 / (1 + hops)`.
- **Scoring integration:** Add a new scorer that projects hop distance and token usage:
  - `token_term = 1 / (1 + log1p(previous_total_tokens))` when available; default to 1.0 if tokens are unavailable.
  - Combine via weighted average (configurable weights; default equal) into a projected scalar (e.g., `hop_token_score`) emitted alongside legacy metrics.
  - Keep existing localization metrics (precision/recall/F1/hit) unchanged in meaning; do not overwrite their semantic role. The projected scalar can be opted into for optimization/analysis and may become primary later via explicit config.
- **Data flow in runtime:** In `run_localization_task`, after predictions are produced and repo materialized, build the static graph, resolve anchors, compute hop distances, and produce both existing metrics and the new projected scalar. Extend telemetry/eval records to carry the projected scalar and per-prediction hop diagnostics without breaking existing consumers.
- **Fairness vs IC/JC:** Keep the evaluator graph read-only and deterministic so comparisons reflect distance-to-anchor, not traversal policy; use IC only as a reference for minimal structures (raw tree, normalization, store) and avoid importing IC runtime/traversal/mutation/frontier/score history/MCP behavior.

## Risks / Trade-offs

- **Shallow ingestion accuracy** → Early extraction may miss cross-file links or non-Python code; mitigate by starting with minimal file/import/call/contains edges and clearly documenting language coverage, adding stubs for future enrichments.
- **Anchor resolution ambiguity** → Heuristic anchors might map to multiple nodes; mitigate with deterministic tie-breaking (shortest path by canonical path match, then symbol match) and diagnostics showing unresolved anchors.
- **Graph build overhead** → Repo snapshot parsing could add latency; mitigate with minimal parsing, skip vendored/virtualenv paths, and cache graph per task repo path within a run.
- **Metric shift risk** → Introducing a projected scalar could affect downstream expectations; mitigate with feature flag/config to keep legacy metrics primary initially, and emit both projected and legacy metrics in telemetry.
- **Missing nodes for predictions** → Predicted paths absent from the graph could yield `None` distances; mitigate by returning unreachable and surfacing diagnostics rather than fabricating placeholder nodes.
- **Dependency surface** → Relying on `networkx` is acceptable but should avoid pulling heavier IC deps; ensure imports stay local to the new module.

## Open Questions

- What minimal language-agnostic extraction guarantees are required for phase 1, and how should we degrade when tree-sitter-style details are unavailable?
- Should the projected scalar be opt-in via config for the first rollout, and how should weights be set for distance vs. tokens?
- Where to persist hop-distance diagnostics (per-file breakdown) in eval records for later analysis without bloating telemetry?
