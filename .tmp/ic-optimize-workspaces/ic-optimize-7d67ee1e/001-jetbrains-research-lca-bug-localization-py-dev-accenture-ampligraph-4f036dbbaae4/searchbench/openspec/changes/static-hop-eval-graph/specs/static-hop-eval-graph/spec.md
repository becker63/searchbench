## ADDED Requirements

### Requirement: Build static evaluation graph from repo snapshot
The system SHALL construct a read-only directed graph for localization evaluation from the materialized repo snapshot using a harness-local wrapper over `llm-tldr`, creating language-agnostic nodes for files and available symbols/functions plus edges for imports, calls, and declared-in/contains relationships. The graph SHALL omit traversal/runtime state and remain immutable after construction.

#### Scenario: Graph builds deterministically from snapshot
- **WHEN** the evaluator materializes a repo and runs the static graph builder
- **THEN** the resulting graph contains nodes for discovered files and functions/symbols with import/call/declared-in edges, and the graph remains read-only for the duration of the evaluation

### Requirement: Anchor and prediction mapping into graph
The system SHALL canonicalize anchors extracted from issue/task text and predicted file paths, resolve them against graph indexes (file and symbol) using a deterministic extraction order (explicit/backticked paths → slashy path-like tokens → identifier candidates → code-block identifiers), and handle unmatched anchors or predictions by marking them unresolved without fabricating nodes.

#### Scenario: Anchors and predictions resolve or degrade gracefully
- **WHEN** anchors and predicted file paths are provided for evaluation
- **THEN** the evaluator resolves each to graph nodes when available, and any unresolved items are marked as such while evaluation continues

### Requirement: Deterministic hop distance query
The system SHALL compute the minimal unweighted hop distance from any resolved anchor node to each resolved predicted file node using the static graph; unreachable nodes SHALL yield a `None`/infinite distance, and results SHALL be deterministic for a given graph snapshot.

#### Scenario: Hop distance returned per prediction
- **WHEN** hop distances are requested for resolved anchors and predicted files
- **THEN** the evaluator returns the smallest hop count per prediction or a `None`/infinite marker when no path exists, with the same output for repeated queries on the same graph
