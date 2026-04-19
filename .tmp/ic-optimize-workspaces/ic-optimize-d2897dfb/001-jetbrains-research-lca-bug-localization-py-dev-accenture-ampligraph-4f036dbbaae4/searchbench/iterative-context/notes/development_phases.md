# Development Phases

Phase 0 — Graph + Events Exist
- Goal: represent and observe a graph.
- Build: NetworkX graph, RxPY event stream, basic node/edge mutations, event emission on every change.
- Capability: I can see every change to the graph.
- Exit: adding a node emits event; adding an edge emits event; updating a node emits event; events are replayable/loggable.

Phase 1 — Deterministic Loop
- Goal: run a loop that mutates the graph over time.
- Build: iteration concept, step counter, fake expand node, loop structure.
- Capability: I can evolve a graph over time deterministically.
- Exit: same input → same event sequence; loop produces multiple iterations; state evolves across iterations.

Phase 2 — Synthetic Expansion
- Goal: simulate jCodeMunch without integrating it.
- Build: fake expand(node) function, fixture-driven graph expansion, predefined neighbors.
- Capability: I can grow a graph without external dependencies.
- Exit: expansion adds nodes and edges; graph grows over iterations; no external calls required.

Phase 3 — Scoring + Selection
- Goal: decide what to expand next.
- Build: scoring function (even simple), frontier selection, expanded flag.
- Capability: I can prioritize exploration.
- Exit: only some nodes expand; order is deterministic; changing score changes behavior.

Phase 4 — Snapshot Testing as Spec
- Goal: lock behavior as a living document.
- Build: snapshot of graph state, snapshot of event stream, fixture-based tests.
- Capability: I can verify graph evolution over time.
- Exit: graph snapshots stable; event snapshots stable; diffs are readable.

Phase 5 — Partial Data Handling
- Goal: prove the system works with incomplete information.
- Build: placeholder nodes, missing metadata handling, edges to unknown nodes.
- Capability: I can operate under uncertainty.
- Exit: graph tolerates missing nodes; expansion can fill gaps later; scoring tolerates missing data.

Phase 6 — Real Adapter (jCodeMunch)
- Goal: replace fake expansion with real queries.
- Build: adapter layer, tool calls, normalization into your graph.
- Capability: I can connect to real codebases.
- Exit: real nodes flow into graph; expansion works with live data; no architectural changes needed.

Phase 7 — Refinement Loop
- Goal: improve quality of context selection.
- Build: scoring heuristics, expansion policies, stopping conditions.
- Capability: I can improve results without changing architecture.
- Exit: deterministic improvements; clear hooks to tune behavior; measurable gains without structural changes.
