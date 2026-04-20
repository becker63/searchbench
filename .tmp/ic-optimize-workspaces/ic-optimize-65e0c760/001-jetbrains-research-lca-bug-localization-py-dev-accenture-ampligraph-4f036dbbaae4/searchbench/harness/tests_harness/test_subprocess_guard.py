from pathlib import Path

from harness.pipeline.types import ExecutionBackend


def test_disallowed_command_rejected():
    backend = ExecutionBackend()
    repo_root = Path(__file__).resolve().parents[2]
    result = backend.run(["echo", "not-allowed"], cwd=repo_root)
    assert not result.success
    assert result.exit_code != 0
