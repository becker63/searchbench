from __future__ import annotations

import sys

import pytest

from types import SimpleNamespace

from harness.localization.models import LCATask
import run as run_module


def test_auto_approve_requires_max_items(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    # Stub dataset fetch and projection to reach the yes/max-items validation quickly.
    monkeypatch.setattr(run_module, "fetch_hf_localization_dataset", lambda *args, **kwargs: [SimpleNamespace(task_id="t1", identity=SimpleNamespace(task_id=lambda: "t1"))])
    monkeypatch.setattr(
        run_module,
        "_select_tasks",
        lambda tasks, max_items, offset: (tasks, {"selected_count": 1, "offset": offset, "limit": max_items, "total_available": 1, "preview": ["t1"]}),
    )
    monkeypatch.setattr(run_module, "_compute_projection", lambda *args, **kwargs: {"total": {"planned": 0, "hard_cap": 0}, "localization": {}, "writer": {}})
    monkeypatch.setattr(run_module, "_require_confirmation", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    argv = [
        "baseline",
        "--config",
        "py",
        "--split",
        "dev",
        "--yes",
    ]
    with pytest.raises(ValueError):
        run_module.main(argv)
