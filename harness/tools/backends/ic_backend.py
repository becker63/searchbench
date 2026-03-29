from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from iterative_context.server import IterativeContextToolRuntime, list_tools

from ..mcp_adapter import mcp_tool_to_openai_tool, parse_text_content_payload, run_async
from ...utils.openai_schema import OpenAITool


class IterativeContextBackend:
    def __init__(self, repo: str, score_fn: Callable[..., float] | None):
        repo_path = Path(repo).expanduser().resolve(strict=False)
        if not repo_path.exists():
            raise RuntimeError(f"[IC] Repo path does not exist: {repo}")

        self._runtime = IterativeContextToolRuntime(score_fn=score_fn, repo_root=repo_path)
        tools = run_async(list_tools())
        self.tool_specs: list[OpenAITool] = [mcp_tool_to_openai_tool(t) for t in tools]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        result_blocks = run_async(self._runtime.call_tool(name, arguments))
        return parse_text_content_payload(result_blocks)
