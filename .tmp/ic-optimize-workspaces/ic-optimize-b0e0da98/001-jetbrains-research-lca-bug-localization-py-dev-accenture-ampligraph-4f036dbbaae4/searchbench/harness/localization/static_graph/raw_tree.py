from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RawFunction(BaseModel):
    id: str
    name: str
    file: str


class RawEdge(BaseModel):
    source: str  # function id OR file path
    target: str  # symbol/module/function name or file path
    kind: Literal["import", "call", "contains", "declares", "references"] | str
    file: str | None = Field(default=None, description="Path of the file edge was observed in")


class RawFile(BaseModel):
    path: str
    functions: list[RawFunction] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    edges: list[RawEdge] = Field(default_factory=list)


class RawTree(BaseModel):
    files: list[RawFile] = Field(default_factory=list)
