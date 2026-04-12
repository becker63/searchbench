from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from openai.types.chat import (
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, Mapping):
                return dumped
        except Exception:
            pass
    return {
        key: getattr(value, key)
        for key in ("type", "function", "name", "description", "parameters")
        if hasattr(value, key)
    }


def _function_from_tool(tool: Any) -> Mapping[str, Any] | None:
    if isinstance(tool, Mapping):
        func = tool.get("function")
    else:
        func = getattr(tool, "function", None)
    if func is None:
        return None
    return _as_mapping(func)


def build_tool_response(call_id: str, content: str) -> ChatCompletionToolMessageParam:
    return ChatCompletionToolMessageParam(
        role="tool",
        tool_call_id=call_id,
        content=content,
    )


def parse_arguments(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    if not isinstance(parsed, Mapping):
        raise ValueError("Tool arguments must be an object")
    return dict(parsed)


def validate_tools(tools: Sequence[ChatCompletionToolParam]) -> None:
    for idx, tool in enumerate(tools):
        tool_map = _as_mapping(tool)
        tool_type = tool_map.get("type")
        if tool_type != "function":
            raise ValueError(f"Tool {idx} must have type='function'")
        func_map = _function_from_tool(tool)
        if not func_map:
            raise ValueError(f"Tool {idx} missing function definition")
        name = func_map.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Tool {idx} function.name must be a non-empty string")
        parameters = func_map.get("parameters")
        if parameters is None:
            continue
        if not isinstance(parameters, Mapping):
            raise ValueError(f"Tool {name} parameters must be a mapping if provided")
        schema_type = cast(str | None, parameters.get("type"))
        if schema_type is None:
            raise ValueError(f"Tool {name} parameters must declare type='object'")
        if schema_type != "object":
            raise ValueError(
                f"Tool {name} parameters.type must be 'object' when present"
            )
        properties = parameters.get("properties")
        if properties is not None and not isinstance(properties, Mapping):
            raise ValueError(
                f"Tool {name} parameters.properties must be a mapping when present"
            )
        strict_flag = func_map.get("strict")
        if strict_flag is None:
            strict_flag = tool_map.get("strict")
        if strict_flag:
            additional_props = parameters.get("additionalProperties")
            if additional_props is not False:
                raise ValueError(
                    f"Tool {name} strict mode requires additionalProperties: false"
                )
            required = parameters.get("required")
            prop_keys = (
                list((properties or {}).keys())
                if isinstance(properties, Mapping)
                else []
            )
            if required is None:
                raise ValueError(
                    f"Tool {name} strict mode requires 'required' to be set"
                )
            if not isinstance(required, list) or any(
                not isinstance(r, str) for r in required
            ):
                raise ValueError(
                    f"Tool {name} strict mode requires 'required' to be a list of strings"
                )
            missing = [k for k in prop_keys if k not in required]
            if missing:
                raise ValueError(
                    f"Tool {name} strict mode requires all properties to be listed in required: {missing}"
                )
