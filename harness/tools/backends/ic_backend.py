from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from iterative_context import exploration
from iterative_context.server import IterativeContextToolRuntime, list_tools

from ..mcp_adapter import mcp_tool_to_openai_tool, parse_text_content_payload, run_async
from ...utils.openai_schema import OpenAITool


class IterativeContextBackend:
    def __init__(self, repo: str, score_fn: Callable[..., float] | None):
        self._runtime = IterativeContextToolRuntime(score_fn=score_fn)
        self._initialize_repo(repo)
        tools = run_async(list_tools())
        self.tool_specs: list[OpenAITool] = [mcp_tool_to_openai_tool(t) for t in tools]

    def _initialize_repo(self, repo: str) -> None:
        repo_path = Path(repo)
        if not repo_path.exists():
            raise RuntimeError(f"[IC] Repo path does not exist: {repo}")
        prev_cwd = Path.cwd()
        try:
            os.chdir(repo_path)
            # NOTE: iterative-context currently loads the active graph from cwd.
            # This fallback relies on that behavior until a repo-specific API is exposed.
            exploration.ensure_graph_loaded()
        finally:
            os.chdir(prev_cwd)

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        result_blocks = run_async(self._runtime.call_tool(name, arguments))
        return parse_text_content_payload(result_blocks)
