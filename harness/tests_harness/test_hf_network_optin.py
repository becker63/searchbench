from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.localization.materialization.hf_materialize import HuggingFaceRepoMaterializer
from harness.localization.materialization.materialize import RepoMaterializationRequest
from harness.telemetry.hosted.hf_lca import fetch_hf_localization_dataset


@pytest.mark.skipif(not os.environ.get("HF_NETWORK_TESTS"), reason="HF network tests are opt-in via HF_NETWORK_TESTS env")
def test_hf_network_smoke_end_to_end(tmp_path):
    """
    Opt-in network smoke: fetch one HF task (py/dev) and materialize its repo at base_sha.
    Skipped by default/CI unless HF_NETWORK_TESTS=1 is set.
    """
    tasks = fetch_hf_localization_dataset(
        "JetBrains-Research/lca-bug-localization",
        dataset_config="py",
        dataset_split="dev",
        revision=os.environ.get("HF_LCA_REVISION"),
        token=os.environ.get("HF_TOKEN"),
    )
    assert tasks, "Expected at least one HF task"
    task = tasks[0]
    mat = HuggingFaceRepoMaterializer(cache_root=tmp_path / "cache")
    req = RepoMaterializationRequest(
        dataset_name=task.identity.dataset_name,
        dataset_config=task.identity.dataset_config,
        dataset_split=task.identity.dataset_split,
        repo_owner=task.identity.repo_owner,
        repo_name=task.identity.repo_name,
        base_sha=task.identity.base_sha,
    )
    result = mat.materialize(req)
    assert result.local_path and Path(result.local_path).exists()
    assert result.checkout_verified is True
