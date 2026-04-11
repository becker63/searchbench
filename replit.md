# Searchbench

A benchmarking and optimization harness for AI agents that navigate codebases. It evaluates and iteratively improves the "policy" (scoring/decision-making logic) of exploratory agents by comparing their performance against baselines using a closed-loop optimization cycle.

## Architecture

- **Package Manager**: `uv` with a workspace layout
- **Language**: Python 3.12+
- **Project Type**: CLI tool (no frontend/web server)

### Package Layout

```
harness/
  entrypoints/        CLI and request models
    cli.py             argparse entry point (run.py delegates here)
    models/
      requests.py      Pydantic request/config models

  orchestration/       State machine and runtime coordination
    machine.py         OptimizationStateMachine, RepairStateMachine
    listeners.py       Tracing listeners for state transitions
    runtime.py         Iteration loop, evaluation wiring, policy pipeline
    types.py           Pydantic models for machine state, context, outcomes

  localization/        Bug localization execution and evaluation
    models.py          Core domain models (LCATask, LCAContext, etc.)
    scoring.py         File-level localization metrics
    errors.py          Typed failure categories
    token_usage.py     Token usage tracking
    telemetry.py       Localization-specific telemetry envelope
    runtime/           Execution and evaluation
      execute.py       Task execution orchestrator
      evaluate.py      Shared evaluation backend (batch scoring)
      records.py       Eval record construction
      agent_runtime.py IC agent iteration runner
    materialization/   Repo cloning and preparation
      materialize.py   Repo materialization
      hf_materialize.py HuggingFace dataset materialization

  backends/            Backend dispatch adapters
    ic.py              Iterative Context backend
    jc.py              JCodeMunch backend
    mcp.py             MCP adapter (tool conversion, async helpers)

  telemetry/           Tracing, scoring, datasets, baselines
    policy_reducer.py  Policy reduction across tasks
    tracing/           Langfuse client and scoring (was langfuse.py)
      __init__.py      Langfuse client wrappers
      score_emitter.py Score emission helpers
      session_policy.py Session-level config
      cerebras_pricing.py Cost tracking for Cerebras
    hosted/            Dataset workflows
      datasets.py      Dataset loading and normalization
      baselines.py     Baseline computation and caching
      experiments.py   Experiment orchestration
      hf_lca.py        HuggingFace LCA loader

  agents/              Agent implementations
    writer.py          Policy generation via LLM
    common.py          Shared agent utilities (usage mapping)

  policy/              Scoring policy management
    current.py         Active scoring function (dynamically updated)
    load.py            Policy loader with validation
    examples.py        Scoring examples for prompt context

  pipeline/            Validation steps (Ruff, Pyright, Pytest)
  prompts/             Prompt templates and models
  utils/               Helpers (diffing, repo resolution, env, etc.)
  tools/               CLI tools (loop_viz, sync_cerebras_models)
```

### Workspace Submodules

- **`iterative-context/`** — Graph-based codebase exploration (cloned directly)
- **`jcodemunch-mcp/`** — MCP adapter for JCodeMunch baseline (cloned directly)

## Backends

- **Iterative Context (IC)**: Graph-based codebase exploration system
- **JCodeMunch (JC)**: Baseline exploration system via MCP

## Key Dependencies

- `openai`, `cerebras-cloud-sdk` — LLM providers
- `langfuse` — Observability/tracing
- `mcp` — Model Context Protocol
- `python-statemachine` — Optimization loop state management
- `pydantic`, `rich`, `pytest`

## Workflow

The app runs as a console workflow: `uv run python run.py --help`

To run experiments, use:
```
uv run python run.py --repo small --symbol expand_node --iterations 3
```

## Setup Notes

- The `requires-python` was lowered from `>=3.13` to `>=3.12` across all workspace members to match available Replit runtime
- `mcp` package was added to `harness/pyproject.toml` as it was an implicit dependency
- Git submodules (`iterative-context`, `jcodemunch-mcp`) are cloned into the workspace directories directly (submodule init is a destructive git op on Replit)
- All workspace dependencies are installed via `uv sync --all-packages`

## Replit Migration Fixes

- **listeners.py**: Fixed all listener methods to accept `model` as a direct kwarg (statemachine 3.0 API change)
- **loop_viz.py**: Fixed `_build_charts()` to provide a `_DummySpan()` as `run_trace`
- **conftest.py**: Added missing `DummyClient`/`DummySpan` methods; patched `_STRICT_DEBUG`
- **iterative-context/pyproject.toml**: Lowered `requires-python` to `>=3.12`

## Environment Variables Needed

- `OPENAI_API_KEY` or `CEREBRAS_API_KEY` — LLM provider
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` — Observability (see `harness/LANGFUSE.md`)
