## Sessions and Scores

- Provide `session_id` when starting root traces/spans to group multi-trace interactions in Langfuse Sessions.
- Child spans created via our helpers inherit the `session_id`; scores emitted via the emitter can include `session_id` for session-level scoring.
- For session-only scores, call the emitter with `session_id` even if no trace/observation IDs are available.

## Test Isolation (Langfuse)

- Tests run with Langfuse credentials cleared and a stubbed client; real network calls to Langfuse must not occur.
- If a test needs to assert the error path, re-monkeypatch `get_langfuse_client` to the stored `_real_get_langfuse_client`.

## Cerebras Pricing Sync

- Custom model definitions (input/output USD per million tokens) to add via Langfuse Models API:
  - Zaizai GLM 4.7 — input $2.25/M, output $2.75/M; regex `(?i)zaizai.*glm.*4\.7`
  - GPT OSS 120B — input $0.35/M, output $0.75/M; regex `(?i)gpt[-_ ]oss[-_ ]120b`
  - Llama 3.1 8B — input $0.10/M, output $0.10/M; regex `(?i)llama[-_ ]3\.1[-_ ]8b`
  - Qwen 3 235B Instruct — input $0.60/M, output $1.20/M; regex `(?i)qwen[-_ ]3[-_ ]235b[-_ ]instruct`
- Use the Models API (`POST /api/public/models`) with the payload from `observability.cerebras_pricing.pricing_payload()`.
- Ensure generations include `model` and usage (when available) so Langfuse can infer costs; otherwise ingest usage manually.
