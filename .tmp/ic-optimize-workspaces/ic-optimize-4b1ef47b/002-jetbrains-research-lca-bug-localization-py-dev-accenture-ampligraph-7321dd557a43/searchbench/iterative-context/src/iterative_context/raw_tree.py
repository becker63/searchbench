from pydantic import BaseModel


class RawFunction(BaseModel):
    id: str
    name: str
    file: str


class RawEdge(BaseModel):
    source: str  # function id OR file path
    target: str  # symbol/module/function name
    kind: str  # "import" | "call"


class RawFile(BaseModel):
    path: str
    functions: list[RawFunction]
    imports: list[str]
    edges: list[RawEdge]


class RawTree(BaseModel):
    files: list[RawFile]
