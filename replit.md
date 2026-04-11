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
- Git submodules (`iterative-context`, `jcodemunch-mcp`) are cloned into the workspace directories directly (submodule init is a destructive git op on Replit)
- All workspace dependencies are installed via `uv sync --all-packages`

## Replit Migration Fixes

- **loop_listeners.py**: Fixed all listener methods (`on_transition`, `on_enter_*`) to accept `model` as a direct keyword argument instead of `event_data.model` — statemachine 3.0 changed the `EventData` API; `model` is now passed as a separate kwarg
- **loop_viz.py**: Fixed `_build_charts()` to provide a `_DummySpan()` as `run_trace` (required by `OptimizationStateMachine._ensure_run_trace`)
- **conftest.py**: Added `start_as_current_observation`, `auth_check`, `shutdown` to `DummyClient`; added `update` to `DummySpan`; patched `_STRICT_DEBUG` to False to prevent test contamination from `run.py`'s `os.environ.setdefault("LANGFUSE_STRICT_DEBUG", "1")`
- **test_experiments_backend.py**: Updated `_run_experiment_task` call to pass required `session_id=None` argument
- **test_loop_state_machine_min.py**: Updated `RecordingSpan` to be a full context manager with `update`/`__enter__`/`__exit__`; fixed `start_observation` helpers to use `@contextmanager`; fixed `run_trace=RecordingSpan()` in optimization test
- **test_repo_resolution_execution.py**: Added `update`, `__enter__`, `__exit__` to `DummySpan`
- **iterative-context/pyproject.toml**: Lowered `requires-python` to `>=3.12` and `ruff` target to `py312`

## Environment Variables Needed

- `OPENAI_API_KEY` or `CEREBRAS_API_KEY` — LLM provider
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` — Observability (see `harness/LANGFUSE.md`)
