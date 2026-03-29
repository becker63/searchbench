from __future__ import annotations

from harness import loop, runner, writer


class DummySpan:
    def __init__(self, name: str = "obs"):
        self.name = name
        self.ended: list[dict[str, object]] = []
        self.scores: list[tuple[str, float]] = []

    def end(self, **kwargs):
        self.ended.append(kwargs)

    def score(self, name: str, value: float | int | bool, metadata=None):
        self.scores.append((name, float(value)))


def test_run_ic_iteration_emits_scores(monkeypatch):
    scores: list[tuple[str, float]] = []
    monkeypatch.setattr(runner, "record_score", lambda span, name, value, metadata=None: scores.append((name, float(value))))
    monkeypatch.setattr(runner, "start_span", lambda *a, **k: DummySpan("span"))
    monkeypatch.setattr(runner, "run_agent", lambda *a, **k: {"observations": [{"tool": "t", "result": {"graph": {"nodes": [1, 2]}}}]})
    monkeypatch.setattr(
        runner,
        "IterativeContextBackend",
        lambda repo, score_fn: type("B", (), {"tool_specs": [], "dispatch": lambda self, name, args: {}})(),
    )

    result = runner.run_ic_iteration({"symbol": "s", "repo": "r"}, score_fn=lambda n, s, c=None: 1.0, steps=1)
    assert result["node_count"] == 2
    assert ("ic.node_count", 2.0) in scores


def test_writer_records_policy_events(monkeypatch):
    attempt_scores: list[tuple[str, float]] = []

    class DummyMessage:
        content = "def score(node, state):\n    return 1.0\n"

    class DummyChoice:
        message = DummyMessage()

    class DummyResponse:
        choices = [DummyChoice()]

    class DummyCompletions:
        def create(self, **kwargs):
            return DummyResponse()

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = DummyChat()

    def fake_client():
        return DummyClient(), "test-model"

    monkeypatch.setattr(writer, "_get_client", fake_client)
    monkeypatch.setattr(writer, "start_span", lambda *a, **k: DummySpan("writer"))
    monkeypatch.setattr(writer, "record_score", lambda obs, name, value, metadata=None: attempt_scores.append((name, float(value))))

    result = writer.generate_policy(
        feedback={"a": 1},
        current_policy="def score(node, state):\n    return 0.0\n",
        tests="",
        scoring_context="",
        feedback_str="",
        guidance_hint="",
        diff_str="",
        diff_hint="",
        comparison_summary="IC vs JC summary",
        parent_trace=None,
    )
    assert "def score" in result
    assert ("writer.policy_compiled", 1.0) in attempt_scores
    assert ("writer.policy_generated", 1.0) in attempt_scores


def test_run_loop_uses_observations(monkeypatch, tmp_path):
    events: list[str] = []
    monkeypatch.setattr(loop, "index_folder", lambda repo: {"repo": repo})
    monkeypatch.setattr(loop, "load_policy", lambda: lambda *_: 1.0)
    monkeypatch.setattr(loop, "run_ic_iteration", lambda *a, **k: {"observations": [{"tool": "x"}], "node_count": 1})
    monkeypatch.setattr(loop, "score", lambda result: {"score": 1.0, "coverage_delta": 1, "ic_nodes": 1, "jc_nodes": 0})
    monkeypatch.setattr(loop, "load_tests", lambda repo: [])
    monkeypatch.setattr(loop, "format_tests_for_prompt", lambda tests: "")
    monkeypatch.setattr(loop, "load_scoring_types", lambda repo: "")
    monkeypatch.setattr(loop, "load_scoring_examples", lambda repo: "")
    monkeypatch.setattr(loop, "format_scoring_context", lambda *a, **k: "")
    monkeypatch.setattr(loop, "_read_policy", lambda *a, **k: "def score(node, state):\n    return 0.0\n")
    monkeypatch.setattr(loop, "_write_policy", lambda *a, **k: None)
    monkeypatch.setattr(loop, "_hash_tests_dir", lambda *a, **k: "hash")
    monkeypatch.setattr(loop, "generate_policy", lambda **kwargs: "def score(node, state):\n    return 1.0\n")

    class DummyPipeline:
        def run(self, repo_root, observation=None):
            return []

    monkeypatch.setattr(loop, "default_pipeline", lambda: DummyPipeline())
    monkeypatch.setattr(loop, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(loop, "start_trace", lambda *a, **k: DummySpan("trace"))
    monkeypatch.setattr(loop, "start_span", lambda *a, **k: DummySpan("span"))
    monkeypatch.setattr(loop, "record_score", lambda *a, **k: events.append("record_score"))
    monkeypatch.setattr(loop, "flush_langfuse", lambda: events.append("flush"))

    baseline = {"jc_result": {"observations": [{"tool": "y"}]}, "jc_metrics": {"jc_nodes": 0, "score": 0.0}}

    history = loop.run_loop({"symbol": "s", "repo": "r"}, iterations=1, baseline_snapshot=baseline)
    assert history and history[0]["score"] == 1.0
    assert "flush" in events
