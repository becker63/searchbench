from __future__ import annotations

import types

import pytest

from harness.telemetry.hosted.hf_lca import (
    HFDatasetLoadError,
    build_langfuse_items_from_hf_tasks,
    compute_lca_task_identity,
    fetch_hf_localization_dataset,
    sync_hf_localization_dataset_to_langfuse,
)


class _FakeDatasets:
    def __init__(self, rows):
        self._rows = rows

    def load_dataset(self, *args, **kwargs):
        return list(self._rows)


def test_hf_loader_normalizes_rows(monkeypatch):
    rows = [
        {
            "repo_owner": "square",
            "repo_name": "okhttp",
            "base_sha": "abc123",
            "issue_title": "t",
            "issue_body": "b",
            "changed_files": ["src/Main.kt", " src/Main.kt "],
            "head_sha": "def456",
            "diff_url": "http://example.com/diff",
            "diff": "diff --git",
            "repo_language": "kotlin",
            "repo_languages": ["kotlin"],
            "repo_license": "apache",
            "repo_stars": 10,
        }
    ]

    fake_module = types.SimpleNamespace(utils=types.SimpleNamespace(AuthenticationError=Exception))
    monkeypatch.setattr(
        "harness.telemetry.hosted.hf_lca._require_dependency",
        lambda: types.SimpleNamespace(load_dataset=_FakeDatasets(rows).load_dataset, utils=fake_module.utils),
    )

    tasks = fetch_hf_localization_dataset(
        "JetBrains-Research/lca-bug-localization", dataset_config="py", dataset_split="dev", revision="main"
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert task.identity.dataset_name == "JetBrains-Research/lca-bug-localization"
    assert task.identity.dataset_config == "py"
    assert task.identity.dataset_split == "dev"
    assert task.identity.repo_owner == "square"
    assert task.identity.repo_name == "okhttp"
    assert task.identity.base_sha == "abc123"
    assert sorted(task.gold.normalized_changed_files()) == ["src/main.kt"]
    assert task.context.repo_language == "kotlin"
    assert task.context.repo_license == "apache"


def test_hf_loader_parses_stringified_changed_files(monkeypatch):
    rows = [
        {
            "repo_owner": "square",
            "repo_name": "okhttp",
            "base_sha": "abc123",
            "issue_title": "t",
            "issue_body": "b",
            "changed_files": "['src/Main.kt']",
        }
    ]

    fake_module = types.SimpleNamespace(utils=types.SimpleNamespace(AuthenticationError=Exception))
    monkeypatch.setattr(
        "harness.telemetry.hosted.hf_lca._require_dependency",
        lambda: types.SimpleNamespace(load_dataset=_FakeDatasets(rows).load_dataset, utils=fake_module.utils),
    )

    tasks = fetch_hf_localization_dataset(
        "JetBrains-Research/lca-bug-localization",
        dataset_config="py",
        dataset_split="dev",
    )

    assert tasks[0].gold.normalized_changed_files() == ["src/main.kt"]


def test_hf_loader_missing_required_fields_raises(monkeypatch):
    rows = [
        {
            "repo_owner": "square",
            "repo_name": "okhttp",
            # base_sha missing
            "changed_files": ["src/Main.kt"],
        }
    ]
    fake_module = types.SimpleNamespace(utils=types.SimpleNamespace(AuthenticationError=Exception))
    monkeypatch.setattr(
        "harness.telemetry.hosted.hf_lca._require_dependency",
        lambda: types.SimpleNamespace(load_dataset=_FakeDatasets(rows).load_dataset, utils=fake_module.utils),
    )
    with pytest.raises(HFDatasetLoadError) as excinfo:
        fetch_hf_localization_dataset("JetBrains-Research/lca-bug-localization")
    assert excinfo.value.category == "schema"


def test_hf_sync_payloads_are_langfuse_items(monkeypatch):
    rows = [
        {
            "repo_owner": "square",
            "repo_name": "okhttp",
            "base_sha": "abc123",
            "issue_title": "t",
            "issue_body": "b",
            "changed_files": ["src/Main.kt"],
        }
    ]

    fake_module = types.SimpleNamespace(utils=types.SimpleNamespace(AuthenticationError=Exception))
    monkeypatch.setattr(
        "harness.telemetry.hosted.hf_lca._require_dependency",
        lambda: types.SimpleNamespace(load_dataset=_FakeDatasets(rows).load_dataset, utils=fake_module.utils),
    )

    tasks = fetch_hf_localization_dataset(
        "JetBrains-Research/lca-bug-localization",
        dataset_config="py",
        dataset_split="dev",
        revision="main",
    )
    items = build_langfuse_items_from_hf_tasks(
        tasks,
        dataset_name="JetBrains-Research/lca-bug-localization",
        dataset_config="py",
        dataset_split="dev",
        revision="main",
    )

    assert items[0]["input"]["identity"]["repo_owner"] == "square"
    assert items[0]["id"] == compute_lca_task_identity(tasks[0])
    assert items[0]["metadata"]["canonical_task_identity"] == items[0]["id"]
    assert items[0]["expected_output"] == {"changed_files": ["src/Main.kt"], "changed_files_count": None, "changed_files_exts": None}
    assert items[0]["metadata"]["source"] == "huggingface"
    assert items[0]["metadata"]["source_config"] == "py"


def test_hf_sync_identity_ignores_source_provenance(monkeypatch):
    rows = [
        {
            "repo_owner": "square",
            "repo_name": "okhttp",
            "base_sha": "abc123",
            "issue_url": "https://example.test/issues/1",
            "issue_title": "t",
            "issue_body": "b",
            "changed_files": ["src/Main.kt"],
        }
    ]
    fake_module = types.SimpleNamespace(utils=types.SimpleNamespace(AuthenticationError=Exception))
    monkeypatch.setattr(
        "harness.telemetry.hosted.hf_lca._require_dependency",
        lambda: types.SimpleNamespace(load_dataset=_FakeDatasets(rows).load_dataset, utils=fake_module.utils),
    )

    task_py = fetch_hf_localization_dataset("hf", dataset_config="py", dataset_split="dev")[0]
    task_java = fetch_hf_localization_dataset("hf", dataset_config="java", dataset_split="test")[0]

    assert task_py.identity.dataset_config != task_java.identity.dataset_config
    assert compute_lca_task_identity(task_py) == compute_lca_task_identity(task_java)


def test_hf_sync_upserts_same_logical_item(monkeypatch):
    rows = [
        {
            "repo_owner": "square",
            "repo_name": "okhttp",
            "base_sha": "abc123",
            "issue_url": "https://example.test/issues/1",
            "issue_title": "t",
            "issue_body": "b",
            "changed_files": ["src/Main.kt"],
        }
    ]
    fake_module = types.SimpleNamespace(utils=types.SimpleNamespace(AuthenticationError=Exception))
    monkeypatch.setattr(
        "harness.telemetry.hosted.hf_lca._require_dependency",
        lambda: types.SimpleNamespace(load_dataset=_FakeDatasets(rows).load_dataset, utils=fake_module.utils),
    )

    class DummyClient:
        def __init__(self):
            self.items: dict[str, dict[str, object]] = {}

        def upsert_dataset_item(self, **kwargs):
            self.items[str(kwargs["id"])] = dict(kwargs)

    client = DummyClient()
    monkeypatch.setattr("harness.telemetry.hosted.hf_lca.get_langfuse_client", lambda: client)

    first = sync_hf_localization_dataset_to_langfuse(
        hf_dataset="hf",
        langfuse_dataset="lca",
        dataset_config="py",
        dataset_split="dev",
    )
    second = sync_hf_localization_dataset_to_langfuse(
        hf_dataset="hf",
        langfuse_dataset="lca",
        dataset_config="py",
        dataset_split="dev",
    )

    assert first[0]["id"] == second[0]["id"]
    assert list(client.items) == [str(first[0]["id"])]
    assert client.items[str(first[0]["id"])]["dataset_name"] == "lca"


def test_hf_sync_requires_langfuse_upsert(monkeypatch):
    rows = [
        {
            "repo_owner": "square",
            "repo_name": "okhttp",
            "base_sha": "abc123",
            "issue_url": "https://example.test/issues/1",
            "issue_title": "t",
            "issue_body": "b",
            "changed_files": ["src/Main.kt"],
        }
    ]
    fake_module = types.SimpleNamespace(utils=types.SimpleNamespace(AuthenticationError=Exception))
    monkeypatch.setattr(
        "harness.telemetry.hosted.hf_lca._require_dependency",
        lambda: types.SimpleNamespace(load_dataset=_FakeDatasets(rows).load_dataset, utils=fake_module.utils),
    )
    monkeypatch.setattr(
        "harness.telemetry.hosted.hf_lca.get_langfuse_client",
        lambda: types.SimpleNamespace(create_dataset_item=lambda **_: None),
    )

    with pytest.raises(HFDatasetLoadError) as excinfo:
        sync_hf_localization_dataset_to_langfuse(
            hf_dataset="hf",
            langfuse_dataset="lca",
            dataset_config="py",
            dataset_split="dev",
        )

    assert excinfo.value.category == "langfuse-client"
