from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from iterative_context.server import IterativeContextToolRuntime, list_tools

from .mcp import mcp_tool_to_openai_tool, parse_text_content_payload, run_async
from harness.utils.openai_schema import OpenAITool


class BackendInit(BaseModel):
    """Validated initialization payload for IC backend."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    repo: str
    score_fn: Callable[..., float] | None


class ToolCallPayload(BaseModel):
    """Validated tool dispatch payload."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any]


class IterativeContextBackend:
    def __init__(self, repo: str, score_fn: Callable[..., float] | None):
        try:
            init_payload = BackendInit(repo=repo, score_fn=score_fn)
        except ValidationError as exc:
            raise RuntimeError(f"[IC] Invalid backend init payload: {exc}") from exc
        repo = init_payload.repo
        repo_path = Path(repo).expanduser().resolve(strict=False)
        if not repo_path.exists():
            raise RuntimeError(f"[IC] Repo path does not exist: {repo}")

        self._runtime = IterativeContextToolRuntime(score_fn=init_payload.score_fn, repo_root=repo_path)
        tools = run_async(list_tools())
        self.tool_specs: list[OpenAITool] = [mcp_tool_to_openai_tool(t) for t in tools]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        payload = ToolCallPayload(name=name, arguments=arguments or {})
        result_blocks = run_async(self._runtime.call_tool(payload.name, payload.arguments))
        return parse_text_content_payload(result_blocks)
