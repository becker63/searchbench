import pytest
from pathlib import Path

from harness.localization.models import (
    LCATaskIdentity,
    LCATask,
    LCAGold,
    LCAContext,
    LocalizationDatasetInfo,
    LocalizationEvalRecord,
    LocalizationEvidence,
    LocalizationEvidenceItem,
    LocalizationGold,
    LocalizationPrediction,
    LocalizationRepoInfo,
    LocalizationRunResult,
    LocalizationTelemetryEnvelope,
    canonicalize_paths,
)
from harness.scoring import ScoreContext, ScoreEngine, summarize_task_score
from harness.localization.runtime.records import build_localization_score_eval_record
from harness.localization.telemetry import build_localization_telemetry
from harness.localization.runtime.execute import run_localization_task
from harness.localization.errors import LocalizationEvaluationError
import harness.localization.runtime.execute as executor_module
import harness.agents.localizer as runner


def _runner_result(
    task: LCATask,
    predicted_files: list[str] | None = None,
    reasoning: str | None = None,
    observations: list[dict[str, object]] | None = None,
    source: str | None = None,
    backend: str | None = None,
) -> LocalizationRunResult:
    return LocalizationRunResult(
        task=task,
        prediction=LocalizationPrediction(
            predicted_files=predicted_files or [],
            reasoning=reasoning,
        ),
        observations=observations or [],
        source=source,
        backend=backend,
    )


def _runner_task(repo: str) -> LCATask:
    return LCATask(
        identity=LCATaskIdentity(
            dataset_name="lca",
            dataset_config="py",
            dataset_split="dev",
            repo_owner="o",
            repo_name="r",
            base_sha="abc",
        ),
        context=LCAContext(issue_title="bug", issue_body="details"),
        gold=LCAGold(changed_files=["a.py"]),
        repo=repo,
    )


def test_localization_run_result_separates_prediction_from_execution() -> None:
    fields = LocalizationRunResult.model_fields
    assert "prediction" in fields
    assert "predicted_files" not in fields
    assert "reasoning" not in fields
    assert set(LocalizationPrediction.model_fields) >= {"predicted_files", "reasoning"}


def test_identity_is_deterministic():
    identity = LCATaskIdentity(
        dataset_name="lca-bug-localization",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="square",
        repo_name="okhttp",
        base_sha="abc123",
        issue_url="https://github.com/square/okhttp/issues/1",
    )
    assert (
        identity.task_id()
        == "lca-bug-localization:py:dev:square/okhttp@abc123:https://github.com/square/okhttp/issues/1"
    )


def test_path_normalization_variants_collapse():
    variants = ["src/foo.py", "./src/foo.py", " SRC\\foo.py "]
    assert canonicalize_paths(variants) == ["src/foo.py"]


def test_eval_builder_is_model_first():
    identity = LCATaskIdentity(
        dataset_name="d",
        dataset_config="c",
        dataset_split="s",
        repo_owner="o",
        repo_name="n",
        base_sha="sha",
        issue_url="url",
    )
    pred = LocalizationPrediction(predicted_files=["a.py"])
    gold = LCAGold(changed_files=["a.py"])
    evidence = LocalizationEvidence(items=[LocalizationEvidenceItem(file="a.py", hints=["line 1"])])
    score_context = ScoreContext(
        predicted_files=("a.py",),
        gold_files=("a.py",),
        gold_min_hops=0,
        issue_min_hops=0,
        per_prediction_hops={"a.py": 0},
    )
    score_bundle = ScoreEngine().evaluate(score_context)
    record = build_localization_score_eval_record(
        identity,
        pred,
        gold,
        score_context=score_context,
        score_bundle=score_bundle,
        evidence=evidence,
        repo_path=Path("/tmp/repo"),
    )
    assert isinstance(record, LocalizationEvalRecord)
    assert isinstance(record.dataset, LocalizationDatasetInfo)
    assert isinstance(record.repo, LocalizationRepoInfo)
    assert isinstance(record.prediction, LocalizationPrediction)
    assert isinstance(record.gold, LocalizationGold)
    assert isinstance(record.evidence, LocalizationEvidence)
    assert record.score_bundle.composed_score == 1.0
    assert record.repo_path == str(Path("/tmp/repo"))
    dumped = record.model_dump()
    assert dumped["dataset"]["name"] == "d"
    assert dumped["prediction"]["predicted_files"] == ["a.py"]
    assert dumped["gold"]["changed_files"] == ["a.py"]
    assert dumped["evidence"]["items"][0]["file"] == "a.py"
    assert dumped["evidence"]["items"][0]["hints"] == ["line 1"]


def test_telemetry_builder_is_model_first():
    identity = LCATaskIdentity(
        dataset_name="d",
        dataset_config="c",
        dataset_split="s",
        repo_owner="o",
        repo_name="n",
        base_sha="sha",
    )
    score_bundle = ScoreEngine().evaluate(ScoreContext(gold_min_hops=0, issue_min_hops=0))
    score_summary = summarize_task_score("task", score_bundle)
    evidence = LocalizationEvidence(items=[LocalizationEvidenceItem(file="a.py", hints=["hint"])])
    envelope = build_localization_telemetry(
        identity,
        score_summary=score_summary,
        changed_files_count=1,
        dataset_source="langfuse",
        repo_language="py",
        repo_license="mit",
        evidence=evidence,
        materialization_events=["clone", "checkout"],
    )
    assert isinstance(envelope, LocalizationTelemetryEnvelope)
    assert isinstance(envelope.dataset, LocalizationDatasetInfo)
    assert isinstance(envelope.repo, LocalizationRepoInfo)
    assert isinstance(envelope.evidence, LocalizationEvidence)
    assert envelope.materialization_events is not None
    dumped = envelope.model_dump()
    assert dumped["score_summary"]["composed_score"] == 1.0
    assert dumped["changed_files_count"] == 1
    assert dumped["dataset_source"] == "langfuse"
    assert dumped["evidence"]["items"][0]["file"] == "a.py"
    assert dumped["evidence"]["items"][0]["hints"] == ["hint"]
    assert dumped["materialization_events"]["events"] == ["clone", "checkout"]


def test_localization_models_reject_unknown_fields():
    with pytest.raises(Exception):
        LocalizationRepoInfo(owner="o", name="n", base_sha="sha", unexpected="x")  # type: ignore[arg-type]


def test_localization_task_uses_runner_predictions(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    gold = LCAGold(changed_files=["gold.py"])
    lca_task = executor_module.LCATask.model_validate(
        {
            "identity": identity.model_dump(),
            "context": LCAContext(issue_title="bug", issue_body="body").model_dump(),
            "gold": gold.model_dump(),
            "repo": str(repo_dir),
        }
    )

    from contextlib import contextmanager

    @contextmanager
    def span_cm():
        yield type("S", (), {"id": "span", "metadata": {}, "end": lambda self, **kw: None})()

    monkeypatch.setattr(executor_module, "start_observation", lambda *a, **k: span_cm())
    prediction, score_bundle, _evidence, _mat, _usage = run_localization_task(
        lca_task,
        runner=lambda task, *_args: _runner_result(task, ["predicted.py"], source="runner"),
    )
    assert prediction.predicted_files == ["predicted.py"]
    assert score_bundle.composed_score is None


def test_localization_execute_attaches_metadata(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    gold = LCAGold(changed_files=[" ./SRC/foo.py "])
    lca_task = executor_module.LCATask.model_validate(
        {
            "identity": identity.model_dump(),
            "context": LCAContext(issue_title="bug", issue_body="body").model_dump(),
            "gold": gold.model_dump(),
            "repo": str(repo_dir),
        }
    )

    from contextlib import contextmanager

    updates: list[dict[str, object]] = []

    @contextmanager
    def span_cm():
        span = type(
            "S",
            (),
            {"id": "span", "metadata": {}, "update": lambda self, **kw: updates.append(kw), "end": lambda self, **kw: None},
        )()
        yield span

    monkeypatch.setattr(executor_module, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(executor_module, "emit_score_for_handle", lambda *a, **k: None)
    prediction, score_bundle, *_ = run_localization_task(
        lca_task,
        runner=lambda task, *_args: _runner_result(task, ["SRC\\foo.py"], source="runner"),
    )
    assert prediction.predicted_files == ["SRC\\foo.py"]
    assert updates
    metadata = updates[-1].get("metadata")
    assert metadata is not None
    assert metadata.get("predicted_files") == ["src/foo.py"]
    assert metadata.get("changed_files") == ["src/foo.py"]
    assert isinstance(metadata.get("telemetry"), dict)
    assert "gold_hop" in score_bundle.results


def test_localization_task_emits_backend_prefixed_score_bundle(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    lca_task = executor_module.LCATask.model_validate(
        {
            "identity": identity.model_dump(),
            "context": LCAContext(issue_title="bug", issue_body="body").model_dump(),
            "gold": LCAGold(changed_files=["a.py"]).model_dump(),
            "repo": str(repo_dir),
        }
    )
    bundle = ScoreEngine().evaluate(
        ScoreContext(
            predicted_files=("a.py",),
            gold_files=("a.py",),
            gold_min_hops=0,
            issue_min_hops=0,
        )
    )

    from contextlib import contextmanager

    emitted: list[dict[str, object]] = []

    @contextmanager
    def span_cm():
        span = type(
            "S",
            (),
            {
                "id": "span",
                "metadata": {},
                "update": lambda self, **kw: None,
                "end": lambda self, **kw: None,
            },
        )()
        yield span

    monkeypatch.setattr(executor_module, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(executor_module, "_score_task", lambda *a, **k: bundle)
    monkeypatch.setattr(
        executor_module,
        "emit_score_for_handle",
        lambda handle, **kwargs: emitted.append(kwargs),
    )

    run_localization_task(
        lca_task,
        runner=lambda task, *_args: _runner_result(task, ["a.py"], backend="ic"),
    )

    names = {item["name"] for item in emitted}
    assert {"ic.composed_score", "ic.gold_hop", "ic.issue_hop"} <= names
    assert all("score_id" not in item for item in emitted)


def test_localization_task_scores_runner_usage_as_visible_component(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo"
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "app.py").write_text("def app():\n    return 1\n")
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    lca_task = executor_module.LCATask.model_validate(
        {
            "identity": identity.model_dump(),
            "context": LCAContext(issue_title="app bug", issue_body="src app").model_dump(),
            "gold": LCAGold(changed_files=["src/app.py"]).model_dump(),
            "repo": str(repo_dir),
        }
    )

    from contextlib import contextmanager

    emitted: list[dict[str, object]] = []

    @contextmanager
    def span_cm():
        span = type(
            "S",
            (),
            {
                "id": "span",
                "metadata": {},
                "update": lambda self, **kw: None,
                "end": lambda self, **kw: None,
            },
        )()
        yield span

    monkeypatch.setattr(executor_module, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(
        executor_module,
        "emit_score_for_handle",
        lambda handle, **kwargs: emitted.append(kwargs),
    )

    _prediction, score_bundle, _evidence, _mat, usage = run_localization_task(
        lca_task,
        runner=lambda task, *_args: LocalizationRunResult(
            task=task,
            prediction=LocalizationPrediction(predicted_files=["src/app.py"]),
            backend="ic",
            raw={"usage_details": {"input": 10, "output": 5, "total": 15}},
        ),
    )

    assert usage.available is True
    assert score_bundle.results["token_efficiency"].available is True
    assert score_bundle.results["token_efficiency"].value is not None
    names = {item["name"] for item in emitted}
    assert "ic.token_efficiency" in names


def test_localization_langfuse_boundary_node_count_visible_without_malformed_bundle_fallback(
    monkeypatch, tmp_path
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    lca_task = executor_module.LCATask.model_validate(
        {
            "identity": identity.model_dump(),
            "context": LCAContext(issue_title="bug", issue_body="body").model_dump(),
            "gold": LCAGold(changed_files=["a.py"]).model_dump(),
            "repo": str(repo_dir),
        }
    )
    bundle = ScoreEngine().evaluate(
        ScoreContext(
            predicted_files=("a.py",),
            gold_files=("a.py",),
            gold_min_hops=0,
            issue_min_hops=0,
        )
    )

    from contextlib import contextmanager

    handle_scores: list[dict[str, object]] = []
    create_score_calls: list[dict[str, object]] = []

    class TaskSpan:
        id = "task-observation"
        metadata: dict[str, object] = {}

        def update(self, **kwargs):
            pass

        def end(self, **kwargs):
            pass

    class BackendSpan:
        id = "backend-observation"

        def score(self, **kwargs):
            handle_scores.append(kwargs)

    class DummyClient:
        def auth_check(self):
            return True

        def create_score(self, **kwargs):
            create_score_calls.append(kwargs)

    @contextmanager
    def span_cm():
        yield TaskSpan()

    def runner_with_visible_node_count(task: LCATask, *_args):
        executor_module.emit_score_for_handle(
            BackendSpan(),
            name="ic.node_count",
            value=2.0,
            data_type="NUMERIC",
        )
        return _runner_result(task, ["a.py"], backend="ic")

    monkeypatch.setattr(executor_module, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(executor_module, "_score_task", lambda *a, **k: bundle)
    monkeypatch.setattr(
        "harness.telemetry.tracing.get_langfuse_client", lambda: DummyClient()
    )

    run_localization_task(lca_task, runner=runner_with_visible_node_count)

    assert [score["name"] for score in handle_scores] == ["ic.node_count"]
    assert create_score_calls == []


def test_run_ic_iteration_strips_gold(monkeypatch, tmp_path):
    captured: list[dict[str, object]] = []

    def fake_run_agent(agent_task: dict[str, object], *args, **kwargs):
        captured.append(agent_task)
        return {"observations": [{"role": "assistant", "content": {"predicted_files": ["src/app.py"], "reasoning": "test"}}]}

    monkeypatch.setattr(
        runner,
        "_finalize_localization",
        lambda task, obs, parent_trace=None: _runner_result(
            task, ["src/app.py"], reasoning="r", observations=obs, source="finalize"
        ),
    )

    monkeypatch.setattr(runner, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        runner,
        "IterativeContextBackend",
        lambda repo, score_fn: type("B", (), {"tool_specs": [], "dispatch": lambda self, name, args: {}})(),
    )
    from contextlib import contextmanager

    @contextmanager
    def span_cm():
        yield type("S", (), {"update": lambda self, **kw: None, "end": lambda self, **kw: None, "id": "span"})()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    result = runner.run_ic_iteration(
        _runner_task(str(repo_dir)),
        score_fn=None,
        steps=1,
        parent_trace=object(),
    )
    assert captured
    assert "changed_files" not in captured[0]
    assert result.prediction.predicted_files == ["src/app.py"]
    assert not hasattr(result, "predicted_files")


def test_run_ic_iteration_recovers_when_finalize_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "run_agent",
        lambda *a, **k: {
            "observations": [
                {
                    "role": "assistant",
                    "content": {
                        "predicted_files": ["src/recovered.py"],
                        "reasoning": "from observations",
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        runner,
        "_finalize_localization",
        lambda task, obs, parent_trace=None: _runner_result(
            task, [], observations=obs, source="finalize"
        ),
    )
    monkeypatch.setattr(
        runner,
        "IterativeContextBackend",
        lambda repo, score_fn: type(
            "B", (), {"tool_specs": [], "dispatch": lambda self, name, args: {}}
        )(),
    )
    from contextlib import contextmanager

    @contextmanager
    def span_cm():
        yield type(
            "S",
            (),
            {"update": lambda self, **kw: None, "end": lambda self, **kw: None, "id": "span"},
        )()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    result = runner.run_ic_iteration(
        _runner_task(str(repo_dir)),
        score_fn=None,
        steps=1,
        parent_trace=object(),
    )

    assert result.prediction.predicted_files == ["src/recovered.py"]
    assert result.source == "fallback"
    assert result.prediction.reasoning == "from observations"


def test_run_ic_iteration_recovers_nested_tool_result_candidates(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    nested_path = repo_dir / "SRC" / "Nested.py"
    monkeypatch.setattr(
        runner,
        "run_agent",
        lambda *a, **k: {
            "observations": [
                {
                    "role": "tool",
                    "result": {
                        "matches": [
                            {"file": str(nested_path)},
                            {"metadata": {"relative_path": "./pkg/Other.py"}},
                        ],
                    },
                    "content": "{}",
                }
            ]
        },
    )
    monkeypatch.setattr(
        runner,
        "_finalize_localization",
        lambda task, obs, parent_trace=None: _runner_result(
            task, [], observations=obs, source="finalize"
        ),
    )
    monkeypatch.setattr(
        runner,
        "IterativeContextBackend",
        lambda repo, score_fn: type(
            "B", (), {"tool_specs": [], "dispatch": lambda self, name, args: {}}
        )(),
    )
    from contextlib import contextmanager

    @contextmanager
    def span_cm():
        yield type(
            "S",
            (),
            {"update": lambda self, **kw: None, "end": lambda self, **kw: None, "id": "span"},
        )()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())

    result = runner.run_ic_iteration(
        _runner_task(str(repo_dir)),
        score_fn=None,
        steps=1,
        parent_trace=object(),
    )

    assert result.prediction.predicted_files == ["src/nested.py", "pkg/other.py"]
    assert result.source == "fallback"


def test_run_ic_iteration_recovers_when_finalization_request_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "run_agent",
        lambda *a, **k: {
            "observations": [
                {
                    "role": "tool",
                    "result": {"files": ["src/from_tool.py"]},
                    "content": "{}",
                }
            ]
        },
    )

    def raise_finalization_error(*_args, **_kwargs):
        raise RuntimeError("queue_exceeded")

    monkeypatch.setattr(runner, "_finalize_localization", raise_finalization_error)
    monkeypatch.setattr(
        runner,
        "IterativeContextBackend",
        lambda repo, score_fn: type(
            "B", (), {"tool_specs": [], "dispatch": lambda self, name, args: {}}
        )(),
    )
    from contextlib import contextmanager

    @contextmanager
    def span_cm():
        yield type(
            "S",
            (),
            {"update": lambda self, **kw: None, "end": lambda self, **kw: None, "id": "span"},
        )()

    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    result = runner.run_ic_iteration(
        _runner_task(str(repo_dir)),
        score_fn=None,
        steps=1,
        parent_trace=object(),
    )

    assert result.prediction.predicted_files == ["src/from_tool.py"]
    assert result.source == "fallback"


def test_candidate_file_collection_normalizes_and_dedupes(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    observations = [
        {
            "role": "tool",
            "result": {
                "files": [
                    str(repo_dir / "SRC" / "App.py"),
                    "./src/app.py",
                    "",
                    "https://example.com/not-a-file.py",
                    "/outside/repo.py",
                    "not_a_file_symbol",
                ],
                "nested": {"owner_path": "pkg/Util.py"},
            },
        },
        {
            "role": "assistant",
            "content": '{"paths": ["pkg/Util.py", "tests/Test_App.py"]}',
        },
    ]

    assert runner._collect_candidate_files_from_observations(  # type: ignore[attr-defined]
        observations,
        repo_root=str(repo_dir),
    ) == ["pkg/util.py", "tests/test_app.py", "src/app.py"]


def test_run_localization_task_empty_finalize_and_empty_fallback_is_typed(
    monkeypatch, tmp_path
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    lca_task = executor_module.LCATask.model_validate(
        {
            "identity": identity.model_dump(),
            "context": LCAContext(issue_title="bug", issue_body="body").model_dump(),
            "gold": LCAGold(changed_files=["a.py"]).model_dump(),
            "repo": str(repo_dir),
        }
    )

    from contextlib import contextmanager

    @contextmanager
    def span_cm():
        yield type(
            "S",
            (),
            {
                "id": "span",
                "metadata": {},
                "update": lambda self, **kw: None,
                "end": lambda self, **kw: None,
            },
        )()

    monkeypatch.setattr(executor_module, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(runner, "run_agent", lambda *a, **k: {"observations": []})
    monkeypatch.setattr(
        runner,
        "_finalize_localization",
        lambda task, obs, parent_trace=None: _runner_result(
            task, [], observations=obs, source="finalize"
        ),
    )
    monkeypatch.setattr(
        runner,
        "IterativeContextBackend",
        lambda repo, score_fn: type(
            "B", (), {"tool_specs": [], "dispatch": lambda self, name, args: {}}
        )(),
    )

    with pytest.raises(LocalizationEvaluationError) as exc_info:
        run_localization_task(lca_task)

    assert exc_info.value.category.value == "runner"
    assert "fallback recovery" in str(exc_info.value)


def test_run_localization_task_preserves_exploration_failure_without_observations(
    monkeypatch, tmp_path
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    identity = LCATaskIdentity(
        dataset_name="lca",
        dataset_config="py",
        dataset_split="dev",
        repo_owner="o",
        repo_name="r",
        base_sha="abc",
    )
    lca_task = executor_module.LCATask.model_validate(
        {
            "identity": identity.model_dump(),
            "context": LCAContext(issue_title="bug", issue_body="body").model_dump(),
            "gold": LCAGold(changed_files=["a.py"]).model_dump(),
            "repo": str(repo_dir),
        }
    )

    from contextlib import contextmanager

    @contextmanager
    def span_cm():
        yield type(
            "S",
            (),
            {
                "id": "span",
                "metadata": {},
                "update": lambda self, **kw: None,
                "end": lambda self, **kw: None,
            },
        )()

    monkeypatch.setattr(executor_module, "start_observation", lambda *a, **k: span_cm())
    monkeypatch.setattr(runner, "start_observation", lambda *a, **k: span_cm())

    def raise_exploration_error(*_args, **_kwargs):
        raise RuntimeError("queue_exceeded")

    monkeypatch.setattr(runner, "run_agent", raise_exploration_error)
    monkeypatch.setattr(
        runner,
        "_finalize_localization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("finalization should not run without observations")
        ),
    )
    monkeypatch.setattr(
        runner,
        "IterativeContextBackend",
        lambda repo, score_fn: type(
            "B", (), {"tool_specs": [], "dispatch": lambda self, name, args: {}}
        )(),
    )

    with pytest.raises(LocalizationEvaluationError) as exc_info:
        run_localization_task(lca_task)

    assert exc_info.value.category.value == "runner"
    assert "failed before producing observations" in str(exc_info.value)
    assert "queue_exceeded" in str(exc_info.value)
