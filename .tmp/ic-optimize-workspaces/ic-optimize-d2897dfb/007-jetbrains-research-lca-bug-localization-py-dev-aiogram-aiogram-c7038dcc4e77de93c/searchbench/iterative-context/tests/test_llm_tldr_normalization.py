from pathlib import Path

import pytest

from iterative_context.injest.llm_tldr_adapter import ingest_repo
from iterative_context.normalize import raw_tree_to_graph
from iterative_context.raw_tree import RawTree
from iterative_context.test_helpers.repos import get_repo


@pytest.fixture(params=["small", "medium"])
def repo(request: pytest.FixtureRequest) -> tuple[str, Path]:
    name = request.param
    return name, get_repo(name)


def normalize_raw_tree(tree: RawTree) -> dict[str, int]:
    """Flatten RawTree into simple aggregate counts for early normalization."""
    counts: dict[str, int] = {"files": 0, "functions": 0, "imports": 0}

    for raw_file in tree.files:
        counts["files"] += 1
        counts["functions"] += len(raw_file.functions)
        counts["imports"] += len(raw_file.imports)

    return counts


def test_ingest_produces_files(repo: tuple[str, Path]) -> None:
    name, path = repo

    tree = ingest_repo(path)

    assert tree.files, f"{name} repo ingestion produced no files: {path}"
    assert any(
        raw_file.functions or raw_file.imports for raw_file in tree.files
    ), f"{name} repo ingestion missing functions/imports: {path}"


def test_functions_have_ids(repo: tuple[str, Path]) -> None:
    name, path = repo

    tree = ingest_repo(path)

    for raw_file in tree.files:
        for fn in raw_file.functions:
            assert fn.id, f"{name} function missing id in {raw_file.path}"
            assert fn.name, f"{name} function missing name in {raw_file.path}"
            assert Path(fn.file).exists(), f"{name} function file missing: {fn.file}"


def test_imports_are_strings(repo: tuple[str, Path]) -> None:
    name, path = repo

    tree = ingest_repo(path)

    for raw_file in tree.files:
        for imp in raw_file.imports:
            assert isinstance(imp, str), f"{name} import is not a string: {imp}"
            assert imp, f"{name} import is empty in file {raw_file.path}"


def test_normalization_is_deterministic(repo: tuple[str, Path]) -> None:
    _, path = repo

    tree1 = ingest_repo(path)
    tree2 = ingest_repo(path)

    norm1 = normalize_raw_tree(tree1)
    norm2 = normalize_raw_tree(tree2)

    assert norm1 == norm2


def test_normalization_has_reasonable_size(repo: tuple[str, Path]) -> None:
    name, path = repo

    tree = ingest_repo(path)
    norm = normalize_raw_tree(tree)

    assert norm["files"] > 0, f"{name} normalization has zero files"
    assert norm["functions"] >= 0, f"{name} normalization has negative functions"
    assert norm["imports"] >= 0, f"{name} normalization has negative imports"

    assert norm["functions"] < 50_000, f"{name} normalization too many functions"
    assert norm["imports"] < 50_000, f"{name} normalization too many imports"


def test_edges_exist(repo: tuple[str, Path]) -> None:
    _, path = repo

    tree = ingest_repo(path)

    total_edges = sum(len(f.edges) for f in tree.files)

    assert total_edges > 0, "no edges extracted from repo"


def test_import_edges_valid(repo: tuple[str, Path]) -> None:
    _, path = repo

    tree = ingest_repo(path)

    for raw_file in tree.files:
        for edge in raw_file.edges:
            if edge.kind == "import":
                assert edge.source
                assert edge.target


def test_graph_has_edges(repo: tuple[str, Path]) -> None:
    _, path = repo

    tree = ingest_repo(path)
    graph = raw_tree_to_graph(tree)

    assert len(graph.edges) > 0


def test_edges_deterministic(repo: tuple[str, Path]) -> None:
    _, path = repo

    t1 = ingest_repo(path)
    t2 = ingest_repo(path)

    e1 = [(e.source, e.target, e.kind) for f in t1.files for e in f.edges]
    e2 = [(e.source, e.target, e.kind) for f in t2.files for e in f.edges]

    assert sorted(e1) == sorted(e2)


def test_call_edges_preserved_across_files(repo: tuple[str, Path]) -> None:
    _, path = repo

    tree = ingest_repo(path)
    raw_calls = [
        (e.source, e.target) for f in tree.files for e in f.edges if e.kind == "call"
    ]

    graph = raw_tree_to_graph(tree)
    graph_calls = [
        (src, dst) for src, dst, data in graph.edges(data=True) if data.get("kind") == "calls"
    ]

    assert len(graph_calls) == len(raw_calls)


def test_unknown_nodes_created(repo: tuple[str, Path]) -> None:
    _, path = repo

    tree = ingest_repo(path)
    graph = raw_tree_to_graph(tree)

    function_ids = {fn.id for raw_file in tree.files for fn in raw_file.functions}

    found_unknown = False
    for _, dst, data in graph.edges(data=True):
        if data.get("kind") != "calls":
            continue
        if dst not in function_ids:
            node_data = graph.nodes.get(dst, {})
            assert node_data, f"call target {dst} missing from graph"
            assert node_data.get("type") in {"unknown", "module", "function"}
            if node_data.get("type") == "unknown":
                found_unknown = True

    assert found_unknown
