from __future__ import annotations

from typing import Any

# Core indexing
from jcodemunch_mcp.tools.index_repo import index_repo
from jcodemunch_mcp.tools.index_folder import index_folder
from jcodemunch_mcp.tools.index_file import index_file
from jcodemunch_mcp.tools.list_repos import list_repos

# File + symbol access
from jcodemunch_mcp.tools.get_file_tree import get_file_tree
from jcodemunch_mcp.tools.get_file_outline import get_file_outline
from jcodemunch_mcp.tools.get_file_content import get_file_content
from jcodemunch_mcp.tools.get_symbol import get_symbol, get_symbols

# Search
from jcodemunch_mcp.tools.search_symbols import search_symbols
from jcodemunch_mcp.tools.search_text import search_text
from jcodemunch_mcp.tools.search_columns import search_columns

# Repo overview
from jcodemunch_mcp.tools.get_repo_outline import get_repo_outline

# Dependency + references
from jcodemunch_mcp.tools.find_importers import find_importers
from jcodemunch_mcp.tools.find_references import find_references
from jcodemunch_mcp.tools.check_references import check_references
from jcodemunch_mcp.tools.get_dependency_graph import get_dependency_graph
from jcodemunch_mcp.tools.get_blast_radius import get_blast_radius

# Diff / structure
from jcodemunch_mcp.tools.get_symbol_diff import get_symbol_diff
from jcodemunch_mcp.tools.get_class_hierarchy import get_class_hierarchy
from jcodemunch_mcp.tools.get_related_symbols import get_related_symbols

# Misc
from jcodemunch_mcp.tools.suggest_queries import suggest_queries
from jcodemunch_mcp.tools.get_session_stats import get_session_stats
from jcodemunch_mcp.tools.invalidate_cache import invalidate_cache

# Core bundle
from jcodemunch_mcp.tools.get_context_bundle import get_context_bundle

__all__ = [
    "index_repo",
    "index_folder",
    "index_file",
    "list_repos",
    "get_file_tree",
    "get_file_outline",
    "get_file_content",
    "get_symbol",
    "get_symbols",
    "search_symbols",
    "search_text",
    "search_columns",
    "get_repo_outline",
    "find_importers",
    "find_references",
    "check_references",
    "get_dependency_graph",
    "get_blast_radius",
    "get_symbol_diff",
    "get_class_hierarchy",
    "get_related_symbols",
    "suggest_queries",
    "get_session_stats",
    "invalidate_cache",
    "get_context_bundle",
]
