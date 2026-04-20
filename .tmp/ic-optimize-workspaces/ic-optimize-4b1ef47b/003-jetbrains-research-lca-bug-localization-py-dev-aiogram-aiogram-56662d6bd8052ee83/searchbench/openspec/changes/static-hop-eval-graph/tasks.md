## 1. Prep and references

- [x] 1.1 Re-read iterative-context minimal pieces (`raw_tree.py`, `normalize.py`, `graph_models.py`, `store.py`, `injest/llm_tldr_adapter.py`) to confirm only evaluator-relevant structs/logic are referenced.
- [x] 1.2 Confirm harness dependencies/config (e.g., `networkx` availability) and choose namespace (e.g., `harness/localization/static_graph/`).

## 2. Static graph kernel

- [x] 2.1 Create harness-local data models mirroring RawTree/RawFile/RawFunction/RawEdge with canonical path handling.
- [x] 2.2 Implement normalization to `networkx.DiGraph` with node/edge attributes (file, symbol, type; import/call/declared-in edges).
- [x] 2.3 Implement read-only Store with symbol/file indexes, neighbor lookup, and shortest-path/hop distance query helpers.
- [x] 2.4 Build a small harness-local `llm-tldr` wrapper (evaluation-oriented, read-only) to ingest repo snapshots into RawTree using language-agnostic, deterministic extraction (skip vendored/venv paths; pluggable per language; degrade gracefully to file-level/import edges when details are unavailable).

## 3. Anchors and mapping

- [x] 3.1 Implement deterministic anchor extraction from issue/task text (explicit/backticked paths → slash-containing path-like tokens → CamelCase/snake_case identifiers → code-block identifiers) with canonicalization.
- [x] 3.2 Map anchors and predicted files to graph nodes via Store indexes; handle unresolved items without failing evaluation and without fabricating nodes.

## 4. Scoring integration

- [x] 4.1 Extend localization scoring to compute hop distances per prediction and derive `distance_term`.
- [x] 4.2 Incorporate token usage via `token_term = 1 / (1 + log1p(previous_total_tokens))` with sensible defaults when unavailable.
- [x] 4.3 Combine terms into weighted projected scalar (e.g., `hop_token_score`) emitted alongside legacy precision/recall/F1/hit without overwriting their semantics (make primary only via explicit flag/config).
- [x] 4.4 Thread graph build and scoring through runtime (`harness/localization/runtime/execute.py`, `models.py`, `records.py`), including telemetry output of hop/token diagnostics and unresolved indicators.

## 5. Testing and validation

- [x] 5.1 Unit test graph normalization/indexing and hop-distance queries with small fixture RawTrees.
- [x] 5.2 Unit test anchor extraction and mapping against sample issue texts and repo layouts.
- [x] 5.3 Integration test scoring projection on a temporary repo snapshot with known anchors/predictions to verify composite score and legacy metrics coexist.
- [x] 5.4 Document validation plan (manual run on sample localization task; ensure fallbacks when graph build fails) and update any developer docs if needed.
