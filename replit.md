# Searchbench

A benchmarking and optimization harness for AI agents that navigate codebases. It evaluates and iteratively improves the "policy" (scoring/decision-making logic) of exploratory agents by comparing their performance against baselines using a closed-loop optimization cycle.

## Architecture

- **Package Manager**: `uv` with a workspace layout
- **Language**: Python 3.12+
- **Project Type**: CLI tool (no frontend/web server)

### Workspace Members

- **`harness/`** — Core benchmarking engine (main package)
  - `run.py` — Entry point (CLI with argparse)
  - `loop.py` — Thin bootstrap / dependency wiring layer
  - `loop_machine.py` — State machine orchestration core (`OptimizationStateMachine`, `RepairStateMachine`)
  - `loop_listeners.py` — Extracted tracing listeners (`RepairTracingListener`, `OptimizationTracingListener`)
  - `loop_types.py` — Pydantic models for machine state, context, outcomes
  - `runner.py` — Agent execution and tool call handling
  - `policy.py` — Active scoring policy (dynamically updated)
  - `pipeline/` — Validation steps (Ruff, Pyright, Pytest)
  - `observability/` — Langfuse integration for tracing and experiments
  - `tools/` — Adapters for IC and JC backends
  - `utils/` — Helpers for diffing, repo resolution, etc.

- **`iterative-context/`** — Git submodule (graph-based codebase exploration)
  - Populated via `git submodule update --init --recursive`

- **`jcodemunch-mcp/`** — Git submodule (MCP adapter for JCodeMunch baseline)
  - Currently uses stub package; run `git submodule update --init --recursive` to populate

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
- After cloning, run `git submodule update --init --recursive` to populate submodules

## Environment Variables Needed

- `OPENAI_API_KEY` or `CEREBRAS_API_KEY` — LLM provider
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` — Observability (see `harness/LANGFUSE.md`)
