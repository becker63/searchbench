from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from jcodemunch_mcp.server import call_tool, list_tools
from mcp.types import TextContent, Tool
from pydantic import BaseModel, ConfigDict, ValidationError

from harness.utils.openai_schema import ChatCompletionToolParam
from .mcp import mcp_tool_to_openai_tool, parse_text_content_payload, run_async


class BackendInit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str


class ToolCallPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any]
    source_observations: list[dict[str, Any]] | None = None


class JCodeMunchBackend:
    def __init__(self, repo: str) -> None:
        try:
            init_payload = BackendInit(repo=repo)
        except ValidationError as exc:
            raise RuntimeError(f"[JC] Invalid backend init payload: {exc}") from exc
        list_tools_fn: Callable[[], Awaitable[list[Tool]]] = cast(
            Callable[[], Awaitable[list[Tool]]],
            list_tools,
        )
        tools = run_async(list_tools_fn())
        self.tool_specs: list[ChatCompletionToolParam] = [mcp_tool_to_openai_tool(t) for t in tools]
        self._initialize_repo(init_payload.repo)

    def _initialize_repo(self, repo: str) -> None:
        repo = repo.strip()
        if not repo:
            raise RuntimeError("[JC] Repo path/url is empty")

        if self._is_existing_local_path(repo):
            self._init_local_repo(repo)
            return

        if self._is_github_url(repo) or self._is_owner_repo(repo):
            self._init_remote_repo(repo)
            return

        raise RuntimeError(f"[JC] Unable to initialize repo: {repo}")

    @staticmethod
    def _is_existing_local_path(repo: str) -> bool:
        return Path(repo).expanduser().exists()

    @staticmethod
    def _is_github_url(repo: str) -> bool:
        return bool(re.match(r"^https?://(www\.)?github\.com/[^/\s]+/[^/\s]+$", repo))

    @staticmethod
    def _is_owner_repo(repo: str) -> bool:
        if "\\" in repo or " " in repo or repo.startswith("/"):
            return False
        parts = repo.split("/")
        return len(parts) == 2 and all(parts)

    def _init_local_repo(self, repo: str) -> None:
        try:
            result = run_async(call_tool("resolve_repo", {"path": repo}))
            parsed = parse_text_content_payload(cast(list[TextContent], result))
            if parsed and isinstance(parsed, dict) and not parsed.get("error"):
                return
        except Exception:
            pass

        result = run_async(call_tool("index_folder", {"path": repo, "incremental": True}))
        parsed = parse_text_content_payload(cast(list[TextContent], result))
        if isinstance(parsed, dict) and parsed.get("error"):
            raise RuntimeError(f"[JC] index_folder failed: {parsed['error']}")

    def _init_remote_repo(self, repo: str) -> None:
        result = run_async(call_tool("index_repo", {"url": repo}))
        parsed = parse_text_content_payload(cast(list[TextContent], result))
        if isinstance(parsed, dict) and parsed.get("error"):
            raise RuntimeError(f"[JC] index_repo failed: {parsed['error']}")

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        call_tool_fn: Callable[[str, dict[str, Any]], Awaitable[list[TextContent]]] = (
            cast(
                Callable[[str, dict[str, Any]], Awaitable[list[TextContent]]],
                call_tool,
            )
        )
        source_obs = arguments.get("source_observations") if isinstance(arguments, dict) else None
        payload = ToolCallPayload(name=name, arguments=arguments or {}, source_observations=source_obs)
        result_blocks = run_async(call_tool_fn(payload.name, payload.arguments))
        parsed = parse_text_content_payload(result_blocks)
        if payload.source_observations:
            return {"result": parsed, "observations": payload.source_observations}
        return parsed
