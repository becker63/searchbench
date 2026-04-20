from __future__ import annotations

import pytest

from harness.telemetry.hosted import datasets
from harness.localization.models import LCATaskIdentity


def test_normalize_localization_task_builds_identity_and_defaults():
    task = {
        "repo": "/tmp/repo",
        "identity": {
            "dataset_name": "lca-bug-localization",
            "dataset_config": "py",
            "dataset_split": "dev",
            "repo_owner": "square",
            "repo_name": "okhttp",
            "base_sha": "abc",
        },
        "context": {"issue_title": "bug", "issue_body": "details"},
        "changed_files": ["a.py", "b.py"],
    }
    normalized = datasets.normalize_localization_task(task)
    assert isinstance(normalized.identity, LCATaskIdentity)
    assert normalized.repo == "/tmp/repo"
    assert normalized.changed_files == ["a.py", "b.py"]
    assert normalized.identity.task_id().startswith("lca-bug-localization:")


def test_langfuse_item_contract_reconstructs_task_and_gold():
    task = datasets.task_from_langfuse_dataset_item(
        {
            "id": "item-1",
            "input": {
                "identity": {
                    "dataset_name": "langfuse-dataset",
                    "dataset_config": "py",
                    "dataset_split": "dev",
                    "repo_owner": "square",
                    "repo_name": "okhttp",
                    "base_sha": "abc",
                },
                "context": {"issue_title": "bug", "issue_body": "details"},
            },
            "expected_output": {"changed_files": ["src/Main.kt"]},
        },
        dataset_name="langfuse-dataset",
    )

    assert task.identity.dataset_name == "langfuse-dataset"
    assert task.identity.repo_owner == "square"
    assert task.repo == "square/okhttp"
    assert task.changed_files == ["src/main.kt"]


def test_langfuse_item_contract_preserves_synced_repo():
    task = datasets.task_from_langfuse_dataset_item(
        {
            "id": "item-1",
            "input": {
                "repo": "square/okhttp",
                "identity": {
                    "dataset_name": "langfuse-dataset",
                    "dataset_config": "py",
                    "dataset_split": "dev",
                    "repo_owner": "square",
                    "repo_name": "okhttp",
                    "base_sha": "abc",
                },
                "context": {"issue_title": "bug", "issue_body": "details"},
            },
            "expected_output": {"changed_files": ["src/Main.kt"]},
        },
        dataset_name="langfuse-dataset",
    )

    assert task.repo == "square/okhttp"


def test_langfuse_item_contract_rejects_missing_expected_output():
    with pytest.raises(datasets.LocalizationDatasetLoadError):
        datasets.task_from_langfuse_dataset_item(
            {
                "input": {
                    "repo_owner": "square",
                    "repo_name": "okhttp",
                    "base_sha": "abc",
                }
            },
            dataset_name="langfuse-dataset",
        )


def test_langfuse_item_contract_rejects_missing_repo_identity_with_item_id():
    with pytest.raises(datasets.LocalizationDatasetLoadError) as excinfo:
        datasets.task_from_langfuse_dataset_item(
            {
                "id": "bad-item",
                "input": {
                    "identity": {
                        "dataset_name": "langfuse-dataset",
                        "dataset_config": "py",
                        "dataset_split": "dev",
                        "repo_owner": "",
                        "repo_name": "",
                        "base_sha": "abc",
                    },
                    "context": {"issue_title": "bug", "issue_body": "details"},
                },
                "expected_output": {"changed_files": ["src/Main.kt"]},
            },
            dataset_name="langfuse-dataset",
        )

    message = str(excinfo.value)
    assert "bad-item" in message
    assert "missing required field" in message
    assert "`repo`" in message
    assert "dataset sync schema is incomplete for runtime execution" in message
