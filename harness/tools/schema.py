from __future__ import annotations

from typing import Any, Callable, Dict, TypedDict


class ToolSpec(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Any]


class Tool:
    name: str
    func: Callable[..., Any]
    description: str
    parameters: Dict[str, Any]

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        parameters: Dict[str, Any],
    ) -> None:
        self.name = name
        self.func = func
        self.description = description
        self.parameters = parameters

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
