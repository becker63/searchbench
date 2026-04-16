from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from harness.localization.models import canonicalize_path

from .filter import _is_probably_text, _should_skip_path, is_supported_source_file
from .raw_tree import RawEdge, RawFile, RawFunction, RawTree

try:
    from tldr import extract_file  # type: ignore[reportMissingTypeStubs]
    from tldr.api import get_imports  # type: ignore[reportMissingTypeStubs]
    from tldr.ast_extractor import (  # type: ignore[reportMissingTypeStubs]
        FunctionInfo,
        ImportInfo,
        ModuleInfo,
    )
except Exception as exc:  # pragma: no cover - import guarded at runtime
    extract_file = None
    get_imports = None
    FunctionInfo = None
    ImportInfo = None
    ModuleInfo = None
    _import_error = exc
else:
    _import_error = None

Extracted = "ModuleInfo | Mapping[str, object]"
FunctionLike = "FunctionInfo | Mapping[str, object]"
ImportLike = "ImportInfo | Mapping[str, object]"


def _iter_functions(extracted: object) -> list[object]:
    if ModuleInfo is not None and isinstance(extracted, ModuleInfo):
        return list(extracted.functions or [])
    if isinstance(extracted, Mapping):
        functions_value = extracted.get("functions", [])
        if isinstance(functions_value, list):
            return list(functions_value)
    return []


def _iter_imports(source: object) -> list[object]:
    if ModuleInfo is not None and isinstance(source, ModuleInfo):
        return list(source.imports or [])
    if isinstance(source, list):
        return list(source)
    if isinstance(source, Mapping):
        imports_value = source.get("imports", [])
        if isinstance(imports_value, list):
            return list(imports_value)
    return []


def _iter_call_graph_edges(module: object) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    if ModuleInfo is not None and isinstance(module, ModuleInfo):
        call_graph = getattr(module, "call_graph", None)
        call_map = getattr(call_graph, "calls", None) if call_graph else None
        if isinstance(call_map, dict):
            for src, targets in cast(dict[str, object], call_map).items():
                if isinstance(targets, list):
                    for target in targets:
                        if isinstance(target, str):
                            edges.append((src, target))
    elif isinstance(module, Mapping):
        call_graph = module.get("call_graph")
        if isinstance(call_graph, Mapping):
            calls = call_graph.get("calls")
            if isinstance(calls, Mapping):
                for src, targets in calls.items():
                    if isinstance(targets, list):
                        for target in targets:
                            if isinstance(target, str):
                                edges.append((str(src), target))
    return edges


def _iter_calls(fn: object) -> list[str]:
    calls: list[str] = []
    if FunctionInfo is not None and isinstance(fn, FunctionInfo):
        maybe_calls = getattr(fn, "calls", None)
        if isinstance(maybe_calls, list):
            calls.extend([c for c in maybe_calls if isinstance(c, str)])
        maybe_called_functions = getattr(fn, "called_functions", None)
        if isinstance(maybe_called_functions, list):
            calls.extend([c for c in maybe_called_functions if isinstance(c, str)])
    elif isinstance(fn, Mapping):
        for key in ("calls", "called_functions"):
            maybe_values = fn.get(key)
            if isinstance(maybe_values, list):
                calls.extend([str(c) for c in maybe_values if isinstance(c, str)])
    return calls


def ingest_repo(root: Path) -> RawTree:
    if extract_file is None or get_imports is None:
        raise ImportError(
            "llm-tldr is required for static graph ingestion"
        ) from _import_error

    root = root.resolve()
    files: list[RawFile] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip_path(path):
            continue
        if not is_supported_source_file(path):
            continue

        try:
            if path.stat().st_size > 512 * 1024:
                continue
        except OSError:
            continue

        if not _is_probably_text(path):
            continue

        try:
            relative_path = canonicalize_path(path.relative_to(root).as_posix())
        except ValueError:
            relative_path = canonicalize_path(path.name)

        try:
            extracted = extract_file(str(path))
        except Exception:
            continue

        functions: list[RawFunction] = []
        edges: list[RawEdge] = []
        for fn in _iter_functions(extracted):
            name: str | None = None
            if FunctionInfo is not None and isinstance(fn, FunctionInfo):
                name = fn.name
            elif isinstance(fn, Mapping):
                raw_name = fn.get("name")
                name = str(raw_name) if isinstance(raw_name, str) else None
            if not name:
                continue

            qualified_name = f"{relative_path}:{name}"
            functions.append(
                RawFunction(
                    id=canonicalize_path(qualified_name),
                    name=name,
                    file=relative_path,
                )
            )

            for called in _iter_calls(fn):
                edges.append(
                    RawEdge(
                        source=canonicalize_path(qualified_name),
                        target=called,
                        kind="call",
                        file=relative_path,
                    )
                )

        for src, target in _iter_call_graph_edges(extracted):
            edges.append(
                RawEdge(
                    source=canonicalize_path(f"{relative_path}:{src}"),
                    target=target,
                    kind="call",
                    file=relative_path,
                )
            )

        try:
            imports_raw = cast(list[object], get_imports(str(path)))  # type: ignore[arg-type]
        except Exception:
            imports_raw = []
        imports: list[str] = []
        for imp in _iter_imports(imports_raw):
            module: str | None = None
            if ImportInfo is not None and isinstance(imp, ImportInfo):
                module = imp.module
            elif isinstance(imp, Mapping):
                raw_module = imp.get("module")
                module = str(raw_module) if isinstance(raw_module, str) else None
            if module:
                imports.append(module)
                edges.append(
                    RawEdge(
                        source=relative_path,
                        target=module,
                        kind="import",
                        file=relative_path,
                    )
                )

        files.append(
            RawFile(
                path=relative_path,
                functions=functions,
                imports=imports,
                edges=edges,
            )
        )

    return RawTree(files=files)
