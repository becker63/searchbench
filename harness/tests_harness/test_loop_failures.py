from __future__ import annotations

from harness import loop
from harness.pipeline.types import StepResult


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


def test_pipeline_repair_succeeds(monkeypatch, tmp_path):
    _common_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "_map_steps", lambda steps: [{"success": s.success, "name": s.name, "stdout": s.stdout, "stderr": s.stderr, "exit_code": s.exit_code} for s in steps])
    gen_calls: list[dict[str, object]] = []

    def generate_policy(**kwargs):
        gen_calls.append(kwargs)
        return "def score(node: object, state: object, context: object | None = None) -> float:\n    return 1.0\n"

    monkeypatch.setattr(loop, "generate_policy", generate_policy)

    class DummyPipeline:
        def __init__(self):
            self.calls = 0

        def run(self, repo_root, observation=None):
            self.calls += 1
            if self.calls == 1:
                return [StepResult(name="ruff_fail", success=False, stdout="", stderr="lint", exit_code=1)]
            return [StepResult(name="ruff_fail", success=True, stdout="", stderr="", exit_code=0)]

    monkeypatch.setattr(loop, "default_pipeline", lambda: DummyPipeline())

    policy, results, success, attempts, meta = loop.synthesize_valid_policy(
        repo_root=tmp_path,
        initial_policy="def score(node: object, state: object, context: object | None = None) -> float:\n    return 0.0\n",
        feedback={},
        tests="",
        scoring_context="",
        feedback_str="",
        guidance_hint="",
        diff_str="",
        diff_hint="",
        comparison_summary=None,
        parent_trace=None,
        max_repair_attempts=3,
    )
    assert success is True
    assert attempts == 2
    assert meta.get("repair_attempts") == 1
    assert any("failure_context" in call for call in gen_calls if call.get("failure_context") is not None)


def test_pipeline_repair_exhausted(monkeypatch, tmp_path):
    _common_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "_map_steps", lambda steps: [{"success": s.success, "name": s.name, "stdout": s.stdout, "stderr": s.stderr, "exit_code": s.exit_code} for s in steps])
    monkeypatch.setattr(loop, "classify_results", lambda results: {"lint_errors": "missing"})
    monkeypatch.setattr(loop, "compute_diff", lambda *a, **k: type("D", (), {"regression": False})())
    monkeypatch.setattr(loop, "format_diff", lambda diff: "diff")
    monkeypatch.setattr(loop, "interpret_diff", lambda diff: "hint")
    monkeypatch.setattr(loop, "generate_policy", lambda **kwargs: "def score(node: object, state: object, context: object | None = None) -> float:\n    return 1.0\n")

    class DummyPipeline:
        def run(self, repo_root, observation=None):
            return [StepResult(name="ruff_fail", success=False, stdout="", stderr="lint", exit_code=1)]

    monkeypatch.setattr(loop, "default_pipeline", lambda: DummyPipeline())

    policy, results, success, attempts, meta = loop.synthesize_valid_policy(
        repo_root=tmp_path,
        initial_policy="def score(node: object, state: object, context: object | None = None) -> float:\n    return 0.0\n",
        feedback={},
        tests="",
        scoring_context="",
        feedback_str="",
        guidance_hint="",
        diff_str="",
        diff_hint="",
        comparison_summary=None,
        parent_trace=None,
        max_repair_attempts=loop._MAX_POLICY_REPAIRS,
    )
    assert success is False
    assert attempts == loop._MAX_POLICY_REPAIRS
    assert meta.get("repair_attempts") == loop._MAX_POLICY_REPAIRS


def test_warning_only_steps_do_not_fail(monkeypatch, tmp_path):
    _common_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "_map_steps", lambda steps: [{"success": s.success, "name": s.name, "stdout": s.stdout, "stderr": s.stderr, "exit_code": s.exit_code} for s in steps])
    gen_calls: list[dict[str, object]] = []

    def generate_policy(**kwargs):
        gen_calls.append(kwargs)
        return "def score(node: object, state: object, context: object | None = None) -> float:\n    return 1.0\n"

    monkeypatch.setattr(loop, "generate_policy", generate_policy)

    class WarningPipeline:
        def run(self, repo_root, observation=None):
            return [
                StepResult(name="basedpyright", success=False, stdout="warning only", stderr="reportUnusedParameter", exit_code=0),
                StepResult(name="pytest_iterative_context", success=False, stdout="warning", stderr="", exit_code=0),
                StepResult(name="ruff_fix", success=False, stdout="notice", stderr="", exit_code=0),
            ]

    monkeypatch.setattr(loop, "default_pipeline", lambda: WarningPipeline())

    policy, results, success, attempts, meta = loop.synthesize_valid_policy(
        repo_root=tmp_path,
        initial_policy="def score(node: object, state: object, context: object | None = None) -> float:\n    return 0.0\n",
        feedback={},
        tests="",
        scoring_context="",
        feedback_str="",
        guidance_hint="",
        diff_str="",
        diff_hint="",
        comparison_summary=None,
        parent_trace=None,
        max_repair_attempts=loop._MAX_POLICY_REPAIRS,
    )
    assert success is True
    assert attempts == 1
    assert meta.get("repair_attempts") == 0
    assert not meta.get("pipeline_feedback")
    assert meta.get("failed_step") is None
    assert any(call.get("failure_context") in (None, "") for call in gen_calls)


def test_nonzero_exit_still_fails(monkeypatch, tmp_path):
    _common_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "_map_steps", lambda steps: [{"success": s.success, "name": s.name, "stdout": s.stdout, "stderr": s.stderr, "exit_code": s.exit_code} for s in steps])
    monkeypatch.setattr(loop, "classify_results", lambda results: {"test_failures": "boom"})
    monkeypatch.setattr(loop, "generate_policy", lambda **kwargs: "def score(node: object, state: object, context: object | None = None) -> float:\n    return 1.0\n")

    class FailingPipeline:
        def run(self, repo_root, observation=None):
            return [StepResult(name="pytest_iterative_context", success=False, stdout="", stderr="test failed", exit_code=1)]

    monkeypatch.setattr(loop, "default_pipeline", lambda: FailingPipeline())

    policy, results, success, attempts, meta = loop.synthesize_valid_policy(
        repo_root=tmp_path,
        initial_policy="def score(node: object, state: object, context: object | None = None) -> float:\n    return 0.0\n",
        feedback={},
        tests="",
        scoring_context="",
        feedback_str="",
        guidance_hint="",
        diff_str="",
        diff_hint="",
        comparison_summary=None,
        parent_trace=None,
        max_repair_attempts=2,
    )
    assert success is False
    assert attempts == 2
    assert meta.get("repair_attempts") == 2
    assert meta.get("pipeline_feedback") == {"test_failures": "boom"}


def test_pipeline_mutation_returns_on_disk_policy(monkeypatch, tmp_path):
    _common_stubs(monkeypatch, tmp_path)
    policy_dir = tmp_path / "harness"
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_dir / "policy.py"

    def _write_policy(code: str, path=policy_path):
        policy_path.write_text(code, encoding="utf-8")

    def _read_policy(path=policy_path):
        return policy_path.read_text(encoding="utf-8")

    monkeypatch.setattr(loop, "_write_policy", _write_policy)
    monkeypatch.setattr(loop, "_read_policy", _read_policy)

    monkeypatch.setattr(loop, "generate_policy", lambda **kwargs: "raw_model_policy")
    monkeypatch.setattr(loop, "_map_steps", lambda steps: [{"success": s.success, "name": s.name, "stdout": s.stdout, "stderr": s.stderr, "exit_code": s.exit_code} for s in steps])

    class MutatingPipeline:
        def run(self, repo_root, observation=None):
            # Simulate ruff --fix writing formatted code.
            policy_path.write_text("formatted_policy", encoding="utf-8")
            return [StepResult(name="ruff_fix", success=True, stdout="", stderr="", exit_code=0)]

    monkeypatch.setattr(loop, "default_pipeline", lambda: MutatingPipeline())

    policy, results, success, attempts, meta = loop.synthesize_valid_policy(
        repo_root=tmp_path,
        initial_policy="initial",
        feedback={},
        tests="",
        scoring_context="",
        feedback_str="",
        guidance_hint="",
        diff_str="",
        diff_hint="",
        comparison_summary=None,
        parent_trace=None,
        max_repair_attempts=loop._MAX_POLICY_REPAIRS,
    )
    assert success is True
    assert policy == "formatted_policy"
    assert meta.get("accepted_policy_source") == "post_pipeline_disk"
    assert meta.get("accepted_policy_changed_by_pipeline") is True


def test_run_loop_propagates_post_pipeline_metadata(monkeypatch, tmp_path):
    _common_stubs(monkeypatch, tmp_path)

    def synthesize_valid_policy(**kwargs):
        return (
            "post_pipeline",
            [],
            True,
            1,
            {
                "repair_attempts": 0,
                "max_policy_repairs": 3,
                "pipeline_passed": True,
                "accepted_policy_source": "post_pipeline_disk",
                "accepted_policy_changed_by_pipeline": True,
            },
        )

    monkeypatch.setattr(loop, "synthesize_valid_policy", synthesize_valid_policy)
    monkeypatch.setattr(loop, "generate_policy", lambda **kwargs: "raw")
    history = loop.run_loop(
        {"symbol": "s", "repo": "r"},
        iterations=1,
        baseline_snapshot=loop.BaselineSnapshot(repo="r", symbol="s", jc_result={}, jc_metrics={}),
    )
    assert len(history) == 1
    last = history[-1]
    assert last.accepted_policy
    assert last.accepted_policy.source == "post_pipeline_disk"
    assert last.accepted_policy.changed_by_pipeline is True


def test_failed_candidate_not_marked_as_accepted(monkeypatch, tmp_path):
    _common_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "_map_steps", lambda steps: [{"success": s.success, "name": s.name, "stdout": s.stdout, "stderr": s.stderr, "exit_code": s.exit_code} for s in steps])
    monkeypatch.setattr(loop, "classify_results", lambda results: {"lint_errors": "missing"})
    monkeypatch.setattr(loop, "generate_policy", lambda **kwargs: "raw")

    class FailingPipeline:
        def run(self, repo_root, observation=None):
            return [StepResult(name="ruff_fix", success=False, stdout="", stderr="lint", exit_code=1)]

    monkeypatch.setattr(loop, "default_pipeline", lambda: FailingPipeline())

    policy, results, success, attempts, meta = loop.synthesize_valid_policy(
        repo_root=tmp_path,
        initial_policy="initial",
        feedback={},
        tests="",
        scoring_context="",
        feedback_str="",
        guidance_hint="",
        diff_str="",
        diff_hint="",
        comparison_summary=None,
        parent_trace=None,
        max_repair_attempts=2,
    )
    assert success is False
    assert "accepted_policy_source" not in meta
    assert "accepted_policy_changed_by_pipeline" not in meta


def test_noop_pipeline_returns_same_policy(monkeypatch, tmp_path):
    _common_stubs(monkeypatch, tmp_path)
    policy_dir = tmp_path / "harness"
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_dir / "policy.py"

    def _write_policy(code: str, path=policy_path):
        policy_path.write_text(code, encoding="utf-8")

    def _read_policy(path=policy_path):
        return policy_path.read_text(encoding="utf-8")

    monkeypatch.setattr(loop, "_write_policy", _write_policy)
    monkeypatch.setattr(loop, "_read_policy", _read_policy)
    monkeypatch.setattr(loop, "generate_policy", lambda **kwargs: "raw_model_policy")
    monkeypatch.setattr(loop, "_map_steps", lambda steps: [{"success": s.success, "name": s.name, "stdout": s.stdout, "stderr": s.stderr, "exit_code": s.exit_code} for s in steps])

    class NoopPipeline:
        def run(self, repo_root, observation=None):
            return [StepResult(name="ruff_fix", success=True, stdout="", stderr="", exit_code=0)]

    monkeypatch.setattr(loop, "default_pipeline", lambda: NoopPipeline())

    policy, results, success, attempts, meta = loop.synthesize_valid_policy(
        repo_root=tmp_path,
        initial_policy="initial",
        feedback={},
        tests="",
        scoring_context="",
        feedback_str="",
        guidance_hint="",
        diff_str="",
        diff_hint="",
        comparison_summary=None,
        parent_trace=None,
        max_repair_attempts=loop._MAX_POLICY_REPAIRS,
    )
    assert success is True
    assert policy == "raw_model_policy"
    assert meta.get("accepted_policy_source") == "post_pipeline_disk"
    assert meta.get("accepted_policy_changed_by_pipeline") is False
