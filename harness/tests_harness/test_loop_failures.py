from __future__ import annotations

from harness import loop


class DummySpan:
    def __init__(self, name: str = "obs"):
        self.name = name

    def end(self, **kwargs):
        pass

    def score(self, name: str, value: float | int | bool, metadata=None):
        pass


def _common_stubs(monkeypatch, tmp_path):
    monkeypatch.setattr(loop, "index_folder", lambda repo: {"repo": repo})
    monkeypatch.setattr(loop, "load_policy", lambda: lambda *_: 1.0)
    monkeypatch.setattr(loop, "run_ic_iteration", lambda *a, **k: {"observations": [{"tool": "x"}], "node_count": 1})
    monkeypatch.setattr(loop, "score", lambda result: {"score": 0.5, "coverage_delta": 0, "ic_nodes": 1, "jc_nodes": 0})
    monkeypatch.setattr(loop, "load_tests", lambda repo: [])
    monkeypatch.setattr(loop, "format_tests_for_prompt", lambda tests: "")
    monkeypatch.setattr(loop, "load_scoring_types", lambda repo: "")
    monkeypatch.setattr(loop, "load_scoring_examples", lambda repo: "")
    monkeypatch.setattr(loop, "format_scoring_context", lambda *a, **k: "")
    monkeypatch.setattr(loop, "_read_policy", lambda *a, **k: "def score(node: object, state: object, context: object | None = None) -> float:\n    return 0.0\n")
    monkeypatch.setattr(loop, "_write_policy", lambda *a, **k: None)
    monkeypatch.setattr(loop, "_hash_tests_dir", lambda *a, **k: "hash")
    monkeypatch.setattr(loop, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(loop, "start_trace", lambda *a, **k: DummySpan("trace"))
    monkeypatch.setattr(loop, "start_span", lambda *a, **k: DummySpan("span"))
    monkeypatch.setattr(loop, "record_score", lambda *a, **k: None)
    monkeypatch.setattr(loop, "flush_langfuse", lambda: None)


def test_pipeline_failure_appends_history(monkeypatch, tmp_path):
    _common_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "classify_results", lambda results: {"lint_errors": "missing"})
    monkeypatch.setattr(loop, "compute_diff", lambda *a, **k: type("D", (), {"regression": False})())
    monkeypatch.setattr(loop, "format_diff", lambda diff: "diff")
    monkeypatch.setattr(loop, "interpret_diff", lambda diff: "hint")
    monkeypatch.setattr(loop, "generate_policy", lambda **kwargs: "def score(node: object, state: object, context: object | None = None) -> float:\n    return 1.0\n")

    class DummyPipeline:
        def run(self, repo_root, observation=None):
            return [type("R", (), {"success": False, "name": "ruff_fail", "stdout": "ruff stdout", "stderr": "ruff stderr", "exit_code": 1})()]

    monkeypatch.setattr(loop, "default_pipeline", lambda: DummyPipeline())

    history = loop.run_loop(
        {"symbol": "s", "repo": "r"},
        iterations=1,
        baseline_snapshot=loop.BaselineSnapshot(repo="r", symbol="s", jc_result={}, jc_metrics={}),
    )
    assert len(history) == 1
    entry = history[0]
    assert entry.pipeline_passed is False
    assert entry.pipeline_feedback == {"lint_errors": "missing"}
    assert entry.failed_step == "ruff_fail"
    assert entry.failed_exit_code == 1
    assert entry.failed_summary == "ruff stderr"
    assert entry.repair_attempts == loop._MAX_POLICY_REPAIRS


def test_pipeline_failure_with_writer_error(monkeypatch, tmp_path):
    _common_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "classify_results", lambda results: {"type_errors": "bad"})
    monkeypatch.setattr(loop, "compute_diff", lambda *a, **k: type("D", (), {"regression": False})())
    monkeypatch.setattr(loop, "format_diff", lambda diff: "diff")
    monkeypatch.setattr(loop, "interpret_diff", lambda diff: "hint")

    calls = {"count": 0}

    def sometimes_fail_generate_policy(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return "def score(node: object, state: object, context: object | None = None) -> float:\n    return 0.0\n"
        raise RuntimeError("writer boom")

    monkeypatch.setattr(loop, "generate_policy", sometimes_fail_generate_policy)

    class DummyPipeline:
        def run(self, repo_root, observation=None):
            return [type("R", (), {"success": False, "name": "pytest_iterative_context", "stdout": "", "stderr": "pytest stderr", "exit_code": 1})()]

    monkeypatch.setattr(loop, "default_pipeline", lambda: DummyPipeline())

    history = loop.run_loop(
        {"symbol": "s", "repo": "r"},
        iterations=1,
        baseline_snapshot=loop.BaselineSnapshot(repo="r", symbol="s", jc_result={}, jc_metrics={}),
    )
    assert len(history) == 1
    entry = history[0]
    assert entry.pipeline_passed is False
    assert entry.writer_error
    assert entry.failed_step == "pytest_iterative_context"
    assert entry.failed_exit_code == 1
    assert entry.failed_summary == "pytest stderr"
    assert entry.repair_attempts == loop._MAX_POLICY_REPAIRS
