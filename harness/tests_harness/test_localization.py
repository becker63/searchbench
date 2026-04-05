import pytest
from pathlib import Path

from localization.models import (
    LCATaskIdentity,
    LCAPrediction,
    LCAGold,
    LocalizationDatasetInfo,
    LocalizationEvalRecord,
    LocalizationEvidence,
    LocalizationEvidenceItem,
    LocalizationGold,
    LocalizationMetrics,
    LocalizationPrediction,
    LocalizationRepoInfo,
    LocalizationTelemetryEnvelope,
)
from localization.scoring import score_file_localization
from localization.eval import build_file_localization_eval_record
from localization.telemetry import build_localization_telemetry


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


def test_file_set_scoring_paths_only():
    prediction = LCAPrediction(predicted_files=["a.py", "c.py"])
    gold = LCAGold(changed_files=["a.py", "b.py"])
    metrics = score_file_localization(prediction, gold)
    assert isinstance(metrics, LocalizationMetrics)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.score == metrics.f1


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
    pred = LCAPrediction(predicted_files=["a.py"])
    gold = LCAGold(changed_files=["a.py"])
    evidence = LocalizationEvidence(items=[LocalizationEvidenceItem(file="a.py", hints=["line 1"])])
    record = build_file_localization_eval_record(identity, pred, gold, evidence=evidence, repo_path=Path("/tmp/repo"))
    assert isinstance(record, LocalizationEvalRecord)
    assert isinstance(record.dataset, LocalizationDatasetInfo)
    assert isinstance(record.repo, LocalizationRepoInfo)
    assert isinstance(record.prediction, LocalizationPrediction)
    assert isinstance(record.gold, LocalizationGold)
    assert isinstance(record.evidence, LocalizationEvidence)
    assert isinstance(record.metrics, LocalizationMetrics)
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
    metrics = LocalizationMetrics(precision=1.0, recall=1.0, f1=1.0, hit=1.0, score=1.0)
    evidence = LocalizationEvidence(items=[LocalizationEvidenceItem(file="a.py", hints=["hint"])])
    envelope = build_localization_telemetry(
        identity,
        metrics=metrics,
        changed_files_count=1,
        repo_language="py",
        repo_license="mit",
        evidence=evidence,
        materialization_events=["clone", "checkout"],
    )
    assert isinstance(envelope, LocalizationTelemetryEnvelope)
    assert isinstance(envelope.dataset, LocalizationDatasetInfo)
    assert isinstance(envelope.repo, LocalizationRepoInfo)
    assert isinstance(envelope.evidence, LocalizationEvidence)
    assert isinstance(envelope.metrics, LocalizationMetrics)
    assert envelope.materialization_events is not None
    dumped = envelope.model_dump()
    assert dumped["metrics"]["score"] == 1.0
    assert dumped["changed_files_count"] == 1
    assert dumped["evidence"]["items"][0]["file"] == "a.py"
    assert dumped["evidence"]["items"][0]["hints"] == ["hint"]
    assert dumped["materialization_events"]["events"] == ["clone", "checkout"]


def test_localization_models_reject_unknown_fields():
    with pytest.raises(Exception):
        LocalizationRepoInfo(owner="o", name="n", base_sha="sha", unexpected="x")  # type: ignore[arg-type]
