## Static Hop Evaluator Validation Plan

1. **Manual localization run**
   - Materialize a repo checkout and run `run_localization_task` with a known issue.
   - Verify metrics still report precision/recall/F1/hit and that `hop_token_score`/`distance_term`/`token_term` appear when extraction succeeds.
2. **Hop-distance sanity check**
   - Confirm per-file hop diagnostics show finite hops for resolved predictions and `null` for unresolved ones (no fabricated nodes).
3. **Fallback behavior**
   - Temporarily disable or remove `llm-tldr`; ensure scoring completes with legacy metrics only (projection fields remain null).
4. **Token usage interaction**
   - Run with a runner payload that includes usage details; confirm `token_term` reflects `1 / (1 + log1p(total_tokens))` and that missing usage yields `token_term=None`.
5. **Telemetry inspection**
   - Inspect emitted telemetry/metadata to ensure projected fields are present without overwriting legacy metric semantics.
