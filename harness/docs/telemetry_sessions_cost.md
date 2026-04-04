## Sessions and Scores (Langfuse v4)

- Use `propagate_context(trace_name/session_id/user_id/tags/metadata/version)` once at entry; all observations inherit it.
- Root/child observations are started via `start_observation` (no legacy trace/span helpers).
- Scores can be emitted with observation/trace or just `session_id` for session-only metrics.

## Test Isolation (Langfuse)

- Tests run with Langfuse credentials cleared and a stubbed client; real network calls to Langfuse must not occur.
- If a test needs to assert the error path, re-monkeypatch `get_langfuse_client` to the stored `_real_get_langfuse_client`.

## Cerebras Pricing & Usage

- Custom model definitions (USD per million tokens) to sync to Langfuse Cloud:
  - Zaizai GLM 4.7 — input $2.25/M, output $2.75/M; regex `(?i)zaizai.*glm.*4\.7`
  - GPT OSS 120B — input $0.35/M, output $0.75/M; regex `(?i)gpt[-_ ]oss[-_ ]120b`
  - Llama 3.1 8B — input $0.10/M, output $0.10/M; regex `(?i)llama[-_ ]3\.1[-_ ]8b`
  - Qwen 3 235B Instruct — input $0.60/M, output $1.20/M; regex `(?i)qwen[-_ ]3[-_ ]235b[-_ ]instruct`
- Sync helper: `observability.cerebras_pricing.sync_cerebras_models()` calls `client.api.models.create` for each entry (requires Langfuse Cloud credentials).
- Generations must include `model` and `usage_details` (OpenAI-style prompt/completion/total) so Langfuse can infer cost. If the provider omits usage, supply it explicitly or add an estimator; do not emit zeroed usage.

## Sessions (where and how)

- Session IDs are resolved only at public entrypoints (CLI or hosted experiment wrappers) via session policy, then propagated.
- Defaults: ad hoc CLI → run-scope; hosted dataset experiments → item-scope (with optional deterministic fallback).
- Children inherit session_id from the propagated context; avoid overwriting mid-run.
- Session-only scores are allowed by passing `session_id` directly to the emitter; otherwise use observation handles to keep grouping.

## Tracing ownership (listener-owned)

- Run/root spans are created once by orchestration bootstrap/listener layers before state machine execution; entrypoints may nest under an external parent but do not create iteration/repair roots.
- Iteration/repair spans and high-level metrics are listener/state-machine owned; `loop.py` remains wiring-only and does not create tracing policy or roots.
- Leaf modules (`loop/runner_agent.py`, `writer.py`, pipeline helpers, baselines) emit child spans only; they require a parent span and must fail/guard rather than create new roots unless an explicit, documented no-op is chosen.
- Allowed leaf spans: writer attempts, backend/model/tool calls, usage/cost/latency, pipeline steps, baseline items — all attached to provided parents.
- Disallowed patterns: leaf-created roots, competing iteration/repair spans, split ownership of high-level scores, silent root creation when no parent is provided.
