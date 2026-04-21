"""MCP surface integration tests."""
# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from iterative_context import exploration, server
from iterative_context.exploration import _repo_signature, _set_active_graph
from iterative_context.graph_models import Graph, GraphNode
from iterative_context.server import IterativeContextToolRuntime, call_tool, list_tools
from iterative_context.test_helpers.graph_dsl import build_graph


@pytest.fixture(autouse=True)
def _reset_active_graph() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    exploration._clear_active_graph()
    yield
    exploration._clear_active_graph()


def _make_graph_with_symbols() -> Graph:
    graph = build_graph(
        {
            "nodes": [
                {"id": "A", "kind": "symbol", "state": "pending"},
                {"id": "B", "kind": "symbol", "state": "pending"},
            ],
            "edges": [
                {"source": "A", "target": "B", "kind": "calls"},
            ],
        }
    )
    graph.nodes["A"]["symbol"] = "expand_node"
    graph.nodes["A"]["file"] = "src/iterative_context/expansion.py"
    graph.nodes["B"]["symbol"] = "other_symbol"
    graph.nodes["B"]["file"] = "src/iterative_context/other.py"
    return graph


def _activate_graph_with_symbols() -> None:
    graph = _make_graph_with_symbols()
    signature = _repo_signature(Path.cwd())
    _set_active_graph(graph, repo_root=Path.cwd().resolve(), signature=signature)


@pytest.mark.anyio
async def test_list_tools_definitions() -> None:
    tools = await list_tools()
    names = {tool.name for tool in tools}
    assert names == {"resolve", "expand", "resolve_and_expand"}

    resolve_tool = next(tool for tool in tools if tool.name == "resolve")
    assert resolve_tool.inputSchema["required"] == ["symbol"]
    assert resolve_tool.inputSchema["properties"]["symbol"]["type"] == "string"

    expand_tool = next(tool for tool in tools if tool.name == "expand")
    assert expand_tool.inputSchema["properties"]["depth"]["type"] == "integer"
    assert "node_id" in expand_tool.inputSchema["required"]


@pytest.mark.anyio
async def test_call_tool_resolve_serializes_node() -> None:
    _activate_graph_with_symbols()

    response = await call_tool("resolve", {"symbol": "expand_node"})
    payload = json.loads(response[0].text)

    assert payload["node"]["id"] == "A"
    assert payload["node"]["symbol"] == "expand_node"
    assert "file" in payload["node"]
    assert payload["full_graph"]["nodes"]
    assert payload["score_source"] == "local_policy"


@pytest.mark.anyio
async def test_call_tool_expand_returns_graph() -> None:
    _activate_graph_with_symbols()

    response = await call_tool("expand", {"node_id": "A", "depth": 1})
    payload = json.loads(response[0].text)
    graph = payload["graph"]

    assert graph["metadata"]["expanded_from"] == "A"
    assert graph["metadata"]["depth"] == 1
    assert graph["nodes"]
    assert payload["full_graph"]["nodes"]
    assert payload["score_source"] == "local_policy"


@pytest.mark.anyio
async def test_call_tool_resolve_and_expand() -> None:
    _activate_graph_with_symbols()

    response = await call_tool("resolve_and_expand", {"symbol": "expand_node", "depth": 1})
    payload = json.loads(response[0].text)

    assert payload["graph"]["metadata"]["expanded_from"] == "A"
    assert payload["graph"]["nodes"]
    assert payload["full_graph"]["nodes"]
    assert payload["score_source"] == "local_policy"


@pytest.mark.anyio
async def test_call_tool_unknown_returns_error() -> None:
    _activate_graph_with_symbols()

    response = await call_tool("unknown_tool", {})
    payload = json.loads(response[0].text)

    assert "error" in payload
    assert "unknown" in payload["error"].lower()
    assert payload["score_source"] == "local_policy"


@pytest.mark.anyio
async def test_runtime_uses_injected_score_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    _activate_graph_with_symbols()
    score_calls: list[str] = []

    def score_fn(node: GraphNode, graph: Graph, step: int) -> float:
        score_calls.append(node.id)
        return 1.0

    # If the injected scorer is used, the local policy should never be consulted.
    def fail_local_policy() -> None:  # pragma: no cover - defensive guard
        raise AssertionError("local policy should not be loaded when score_fn is injected")

    monkeypatch.setattr(server, "load_local_policy", fail_local_policy)

    runtime = IterativeContextToolRuntime(score_fn=score_fn)
    response = await runtime.call_tool("expand", {"node_id": "A", "depth": 1})
    payload = json.loads(response[0].text)

    assert score_calls
    assert payload["graph"]["metadata"]["expanded_from"] == "A"
    assert payload["score_source"] == "injected"


@pytest.mark.anyio
async def test_runtime_uses_local_policy_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _activate_graph_with_symbols()
    score_calls: list[str] = []

    def local_score(node: GraphNode, graph: Graph, step: int) -> float:
        score_calls.append(node.id)
        return 2.0

    monkeypatch.setattr(server, "load_local_policy", lambda: local_score)

    runtime = IterativeContextToolRuntime()
    response = await runtime.call_tool("expand", {"node_id": "A", "depth": 1})
    payload = json.loads(response[0].text)

    assert score_calls
    assert payload["score_source"] == "local_policy"


@pytest.mark.anyio
async def test_runtime_falls_back_to_default_when_policy_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _activate_graph_with_symbols()
    score_calls: list[str] = []

    def default_score(node: GraphNode, graph: Graph, step: int) -> float:
        score_calls.append(node.id)
        return 3.0

    monkeypatch.setattr(server, "load_local_policy", lambda: None)
    monkeypatch.setattr(server, "default_score_fn", default_score)

    runtime = IterativeContextToolRuntime()
    response = await runtime.call_tool("expand", {"node_id": "A", "depth": 1})
    payload = json.loads(response[0].text)

    assert score_calls
    assert payload["score_source"] == "default"


def _write_repo(root: Path, symbol: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "module.py").write_text(
        f"def {symbol}() -> str:\n    return '{symbol}'\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def two_repos(tmp_path: Path) -> tuple[Path, Path]:
    repo_a = _write_repo(tmp_path / "repo_a", "alpha_symbol")
    repo_b = _write_repo(tmp_path / "repo_b", "beta_symbol")
    return repo_a, repo_b


def test_explicit_repo_load_without_chdir(two_repos: tuple[Path, Path]) -> None:
    repo_a, _ = two_repos
    start_cwd = Path.cwd()

    exploration.ensure_graph_loaded(repo_root=repo_a)
    node = exploration.resolve("alpha_symbol")

    assert node is not None
    assert exploration.resolve("beta_symbol") is None
    assert Path.cwd() == start_cwd
    assert str(node.id).endswith("alpha_symbol")


@pytest.mark.anyio
async def test_runtime_binds_to_repo_root(two_repos: tuple[Path, Path]) -> None:
    repo_a, _ = two_repos
    runtime = IterativeContextToolRuntime(repo_root=repo_a)

    response = await runtime.call_tool("resolve", {"symbol": "alpha_symbol"})
    payload = json.loads(response[0].text)

    assert payload["node"]["symbol"] == "alpha_symbol"

    missing = await runtime.call_tool("resolve", {"symbol": "beta_symbol"})
    missing_payload = json.loads(missing[0].text)
    assert missing_payload["node"] is None


@pytest.mark.anyio
async def test_runtimes_isolate_repos(two_repos: tuple[Path, Path]) -> None:
    repo_a, repo_b = two_repos
    runtime_a = IterativeContextToolRuntime(repo_root=repo_a)
    runtime_b = IterativeContextToolRuntime(repo_root=repo_b)

    first = await runtime_a.call_tool("resolve", {"symbol": "alpha_symbol"})
    first_payload = json.loads(first[0].text)
    second = await runtime_b.call_tool("resolve", {"symbol": "beta_symbol"})
    second_payload = json.loads(second[0].text)

    expand_a = await runtime_a.call_tool(
        "expand", {"node_id": first_payload["node"]["id"], "depth": 0}
    )
    expand_b = await runtime_b.call_tool(
        "expand", {"node_id": second_payload["node"]["id"], "depth": 0}
    )

    graph_a = json.loads(expand_a[0].text)["graph"]
    graph_b = json.loads(expand_b[0].text)["graph"]

    symbols_a = {n["symbol"] for n in graph_a["nodes"]}
    symbols_b = {n["symbol"] for n in graph_b["nodes"]}

    assert symbols_a == {"alpha_symbol"}
    assert symbols_b == {"beta_symbol"}


@pytest.mark.anyio
async def test_runtime_defaults_to_cwd(
    two_repos: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_a, _ = two_repos
    monkeypatch.chdir(repo_a)

    runtime = IterativeContextToolRuntime()
    response = await runtime.call_tool("resolve", {"symbol": "alpha_symbol"})
    payload = json.loads(response[0].text)

    assert payload["node"]["symbol"] == "alpha_symbol"


def test_ensure_graph_loaded_idempotent(
    two_repos: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_a, _ = two_repos
    calls: list[Path] = []

    original = exploration._build_graph_from_repo

    def _tracked(root: Path) -> Graph:
        calls.append(root)
        return original(root)

    monkeypatch.setattr(exploration, "_build_graph_from_repo", _tracked)

    exploration.ensure_graph_loaded(repo_root=repo_a)
    exploration.ensure_graph_loaded(repo_root=repo_a)

    assert calls == [Path(repo_a).resolve()]
