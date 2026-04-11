from pathlib import Path

import pytest

from harness.orchestration import _write_policy


def test_only_policy_file_can_be_written(tmp_path: Path):
    with pytest.raises(RuntimeError):
        _write_policy("hi", tmp_path / "other.py")
