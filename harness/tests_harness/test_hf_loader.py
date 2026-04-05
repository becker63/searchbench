from __future__ import annotations

import types

import pytest

from harness.observability.hf_lca import HFDatasetLoadError, fetch_hf_localization_dataset


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
        "harness.observability.hf_lca._require_dependency",
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
    assert sorted(task.gold.normalized_changed_files()) == ["src/Main.kt"]
    assert task.context.repo_language == "kotlin"
    assert task.context.repo_license == "apache"


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
        "harness.observability.hf_lca._require_dependency",
        lambda: types.SimpleNamespace(load_dataset=_FakeDatasets(rows).load_dataset, utils=fake_module.utils),
    )
    with pytest.raises(HFDatasetLoadError) as excinfo:
        fetch_hf_localization_dataset("JetBrains-Research/lca-bug-localization")
    assert excinfo.value.category == "schema"
