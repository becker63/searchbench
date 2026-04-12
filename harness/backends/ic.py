from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict, ValidationError

from iterative_context.server import IterativeContextToolRuntime, list_tools

from .mcp import mcp_tool_to_openai_tool, parse_text_content_payload, run_async, serialize_tool_result_for_model
from harness.utils.openai_schema import ChatCompletionToolParam


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
    source_observations: list[dict[str, Any]] | None = None


def _coerce_arguments_for_schema(arguments: dict[str, Any], tool_spec: ChatCompletionToolParam | None) -> dict[str, Any]:
    if not isinstance(arguments, dict) or not isinstance(tool_spec, dict):
        return arguments
    func = tool_spec.get("function")
    if not isinstance(func, dict):
        return arguments
    params = func.get("parameters")
    if not isinstance(params, dict):
        return arguments
    props = params.get("properties")
    if not isinstance(props, dict):
        return arguments
    coerced = dict(arguments)
    for key, schema in props.items():
        if key not in coerced:
            continue
        if isinstance(schema, dict) and schema.get("type") == "integer":
            val = coerced[key]
            if isinstance(val, str):
                try:
                    coerced[key] = int(val)
                except ValueError:
                    pass
    return coerced


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
        self.tool_specs: list[ChatCompletionToolParam] = [mcp_tool_to_openai_tool(t) for t in tools]
        self._tool_spec_by_name: dict[str, ChatCompletionToolParam] = {}
        for spec in self.tool_specs:
            func = spec.get("function") if isinstance(spec, dict) else None
            name = func.get("name") if isinstance(func, dict) else None
            if isinstance(name, str):
                self._tool_spec_by_name[name] = spec

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        source_obs = arguments.get("source_observations") if isinstance(arguments, dict) else None
        payload = ToolCallPayload(name=name, arguments=arguments or {}, source_observations=source_obs)
        coerced_args = _coerce_arguments_for_schema(payload.arguments, self._tool_spec_by_name.get(payload.name))
        result_blocks = run_async(self._runtime.call_tool(payload.name, coerced_args))
        parsed = parse_text_content_payload(result_blocks)
        # Preserve original payload shape for callers/tests; add observations when provided.
        if payload.source_observations:
            return {"result": parsed, "observations": payload.source_observations}
        return parsed
