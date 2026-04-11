from __future__ import annotations

from harness.telemetry import datasets
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
