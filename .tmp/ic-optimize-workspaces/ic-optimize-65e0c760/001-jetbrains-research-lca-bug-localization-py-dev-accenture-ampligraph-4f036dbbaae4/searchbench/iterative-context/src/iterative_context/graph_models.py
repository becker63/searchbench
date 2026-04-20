# iterative-context/graph_models.py

from __future__ import annotations

# Public alias for the directed graph type used throughout the system.
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

import networkx as nx
from pydantic import BaseModel, Field
from rx.subject.subject import Subject

if TYPE_CHECKING:
    Graph = nx.DiGraph[Any]
else:
    Graph = nx.DiGraph

# ----------------------------------------------------------------------
# Core node definitions
# ----------------------------------------------------------------------


class BaseNode(BaseModel):
    """Common fields for all graph nodes."""

    id: str
    label: str | None = None
    kind: Literal["symbol", "function", "file", "type"]


class PendingNode(BaseNode):
    state: Literal["pending"] = "pending"


class PrunedNode(BaseNode):
    state: Literal["pruned"] = "pruned"


class Evidence(BaseModel):
    """Evidence attached to anchor / resolved nodes."""

    snippet: str
    file: str | None = None
    startLine: int | None = None


class AnchorNode(BaseNode):
    state: Literal["anchor"] = "anchor"
    tokens: int | None = None
    evidence: Evidence | None = None


class ResolvedNode(BaseNode):
    state: Literal["resolved"] = "resolved"
    tokens: int
    evidence: Evidence


# Union of all node variants – discriminated by the ``state`` field.
GraphNode = Annotated[
    PendingNode | PrunedNode | AnchorNode | ResolvedNode,
    Field(discriminator="state"),
]


# ----------------------------------------------------------------------
# Edge definition
# ----------------------------------------------------------------------


class GraphEdge(BaseModel):
    """Directed edge between two nodes."""

    id: str | None = None
    source: str
    target: str
    kind: Literal["calls", "imports", "references"]
    primary: bool | None = None


# ----------------------------------------------------------------------
# Event definitions – each variant is discriminated by the ``type`` field.
# ----------------------------------------------------------------------


class AddNodesEvent(BaseModel):
    type: Literal["addNodes"] = "addNodes"
    nodes: Sequence[GraphNode]
    reason: str | None = None


class AddEdgesEvent(BaseModel):
    type: Literal["addEdges"] = "addEdges"
    edges: Sequence[GraphEdge]
    reason: str | None = None


class UpdateNodeEvent(BaseModel):
    type: Literal["updateNode"] = "updateNode"
    id: str
    patch: dict[str, Any]  # Partial GraphNode representation


class IterationEvent(BaseModel):
    type: Literal["iteration"] = "iteration"
    step: int
    description: str | None = None


# Union of all supported events – with discriminator
GraphEvent = Annotated[
    AddNodesEvent | AddEdgesEvent | UpdateNodeEvent | IterationEvent,
    Field(discriminator="type"),
]


def create_event_subject() -> Subject:
    """Create a deterministic RxPY Subject for emitting GraphEvents."""
    return Subject()


__all__ = [
    "GraphNode",
    "GraphEdge",
    "GraphEvent",
    "Graph",
    "create_event_subject",
]
