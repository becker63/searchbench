"""
Static evaluator graph kernel for localization:
- harness-local wrapper over llm-tldr for static extraction
- normalization to a simple DiGraph
- read-only store and hop-distance helpers
- deterministic anchor extraction and mapping
"""

from .anchors import extract_anchors, resolve_anchors, resolve_predictions
from .distance import HopDistanceResult, compute_hop_distance_summary
from .ingest import ingest_repo
from .normalize import raw_tree_to_graph
from .raw_tree import RawEdge, RawFile, RawFunction, RawTree
from .store import GraphStore

__all__ = [
    "extract_anchors",
    "resolve_anchors",
    "resolve_predictions",
    "compute_hop_distance_summary",
    "HopDistanceResult",
    "ingest_repo",
    "raw_tree_to_graph",
    "RawEdge",
    "RawFile",
    "RawFunction",
    "RawTree",
    "GraphStore",
]
