# iterative-context/src/iterative_context/expansion.py

"""Deterministic graph expansion utilities.

This module provides a single function ``expand_node`` that computes the minimal
set of events required to expand a node in the deterministic, event‑driven
graph system described in the project specification.

The function is deliberately simple:
* It creates exactly one child node whose identifier is ``<parent>_child``.
* If that child already exists, no mutation events are emitted for node or edge
  creation – only the parent is marked as resolved.
* Otherwise it emits ``addNodes``, ``addEdges`` and ``updateNode`` events in that
  order.

All events conform to the strict schema defined in ``graph_models.py``.
"""

from __future__ import annotations

from iterative_context.graph_models import (
    AddEdgesEvent,
    AddNodesEvent,
    Graph,
    GraphEdge,
    GraphEvent,
    PendingNode,
    UpdateNodeEvent,
)


def expand_node(
    node_to_expand: dict[str, str],
    graph: Graph,
) -> list[GraphEvent]:
    """
    MUTATION PRIMITIVE.

    This function produces events that MODIFY the graph.

    It does NOT:
    - traverse the graph
    - query neighbors
    - compute neighborhoods

    It only defines how a single node expands.

    This is intentionally minimal and deterministic.

    Produce the deterministic sequence of events that expands ``node_to_expand``.

    Parameters
    ----------
    node_to_expand:
        Mapping with keys ``"id"`` (the node identifier) and ``"state"``.
        The ``state`` value is not inspected by the algorithm – it is included
        for completeness and future extensibility.

    existing_nodes:
        A list containing *all* node identifiers currently present in the graph.

    Returns
    -------
    List[GraphEvent]
        A list of Pydantic event objects ready to be fed to the event‑processing
        pipeline. The order matches the examples in the specification:
        ``addNodes`` → ``addEdges`` → ``updateNode`` (when a new child is
        created), or just ``updateNode`` when the child already exists.

    Notes
    -----
    * The function is pure and deterministic – its output depends solely on
      ``node_to_expand["id"]`` and ``existing_nodes``.
    * No duplicate nodes or edges are ever produced.
    * The parent node is always marked as resolved with ``tokens`` set to ``1``.
    """
    parent_id: str = node_to_expand["id"]
    child_id: str = f"{parent_id}_child"
    existing_nodes = list(graph.nodes)

    events: list[GraphEvent] = []

    if child_id in existing_nodes:
        # Child already present – only mark the parent as resolved.
        events.append(
            UpdateNodeEvent(
                id=parent_id,
                patch={"state": "resolved", "tokens": 1},
            )
        )
        return events

    # Child does not exist – create node, edge, then resolve parent.
    # Create a full GraphNode object for the new child
    child_node = PendingNode(id=child_id, kind="symbol")
    events.append(
        AddNodesEvent(
            nodes=[child_node],
        )
    )
    # Create a full GraphEdge object connecting parent to child
    edge = GraphEdge(source=parent_id, target=child_id, kind="calls")
    events.append(
        AddEdgesEvent(
            edges=[edge],
        )
    )
    events.append(
        UpdateNodeEvent(
            id=parent_id,
            patch={"state": "resolved", "tokens": 1},
        )
    )
    return events


__all__ = ["expand_node"]
