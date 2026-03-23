from __future__ import annotations

from ..schema import Tool
from . import (
    check_references,
    find_importers,
    find_references,
    get_blast_radius,
    get_class_hierarchy,
    get_context_bundle,
    get_dependency_graph,
    get_file_content,
    get_file_outline,
    get_file_tree,
    get_related_symbols,
    get_repo_outline,
    get_session_stats,
    get_symbol,
    get_symbol_diff,
    get_symbols,
    index_file,
    index_folder,
    index_repo,
    invalidate_cache,
    list_repos,
    search_columns,
    search_symbols,
    search_text,
    suggest_queries,
)

JC_TOOLS = [
    Tool(
        "index_repo",
        index_repo,
        "Index a GitHub repository",
        {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    ),
    Tool(
        "index_folder",
        index_folder,
        "Index a local folder",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
    Tool(
        "index_file",
        index_file,
        "Index a single file",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
    Tool(
        "list_repos",
        list_repos,
        "List indexed repositories",
        {"type": "object", "properties": {}},
    ),
    Tool(
        "get_file_tree",
        get_file_tree,
        "Get repository file tree",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    ),
    Tool(
        "get_file_outline",
        get_file_outline,
        "Get symbols in a file",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    ),
    Tool(
        "get_file_content",
        get_file_content,
        "Get file contents",
        {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "file_path": {"type": "string"}},
            "required": ["repo", "file_path"],
        },
    ),
    Tool(
        "get_symbol",
        get_symbol,
        "Get symbol source",
        {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "symbol_id": {"type": "string"}},
            "required": ["repo", "symbol_id"],
        },
    ),
    Tool(
        "get_symbols",
        get_symbols,
        "Get multiple symbols",
        {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "symbol_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["repo", "symbol_ids"],
        },
    ),
    Tool(
        "search_symbols",
        search_symbols,
        "Search symbols",
        {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "query": {"type": "string"}},
            "required": ["repo", "query"],
        },
    ),
    Tool(
        "search_text",
        search_text,
        "Search text",
        {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "query": {"type": "string"}},
            "required": ["repo", "query"],
        },
    ),
    Tool(
        "search_columns",
        search_columns,
        "Search columns",
        {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "query": {"type": "string"}},
            "required": ["repo", "query"],
        },
    ),
    Tool(
        "get_repo_outline",
        get_repo_outline,
        "Get repo overview",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    ),
    Tool(
        "find_importers",
        find_importers,
        "Find file importers",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    ),
    Tool(
        "find_references",
        find_references,
        "Find references",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    ),
    Tool(
        "check_references",
        check_references,
        "Check references",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    ),
    Tool(
        "get_dependency_graph",
        get_dependency_graph,
        "Get dependency graph",
        {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "file": {"type": "string"}},
            "required": ["repo", "file"],
        },
    ),
    Tool(
        "get_blast_radius",
        get_blast_radius,
        "Get blast radius",
        {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "symbol": {"type": "string"}},
            "required": ["repo", "symbol"],
        },
    ),
    Tool(
        "get_symbol_diff",
        get_symbol_diff,
        "Diff symbols",
        {
            "type": "object",
            "properties": {"repo_a": {"type": "string"}, "repo_b": {"type": "string"}},
            "required": ["repo_a", "repo_b"],
        },
    ),
    Tool(
        "get_class_hierarchy",
        get_class_hierarchy,
        "Get class hierarchy",
        {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "class_name": {"type": "string"}},
            "required": ["repo", "class_name"],
        },
    ),
    Tool(
        "get_related_symbols",
        get_related_symbols,
        "Find related symbols",
        {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "symbol_id": {"type": "string"}},
            "required": ["repo", "symbol_id"],
        },
    ),
    Tool(
        "suggest_queries",
        suggest_queries,
        "Suggest queries",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    ),
    Tool(
        "get_session_stats",
        get_session_stats,
        "Get session stats",
        {"type": "object", "properties": {}},
    ),
    Tool(
        "invalidate_cache",
        invalidate_cache,
        "Invalidate cache",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    ),
    Tool(
        "get_context_bundle",
        get_context_bundle,
        "Get context bundle",
        {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "symbol_id": {"type": "string"}},
            "required": ["repo", "symbol_id"],
        },
    ),
]
