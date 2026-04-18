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
    assert task.changed_files == ["src/main.kt"]


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
