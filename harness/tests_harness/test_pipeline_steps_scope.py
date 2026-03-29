from pathlib import Path

from harness.pipeline.steps import RuffFixStep, BasedPyrightStep
from harness.pipeline.types import StepResult, ExecutionBackend


class FakeBackend(ExecutionBackend):
    def __init__(self):
        self.last_cmd: list[str] | None = None

    def run(self, cmd: list[str], cwd: Path, name: str | None = None) -> StepResult:
        self.last_cmd = cmd
        return StepResult(
            name=name or "step",
            success=True,
            stdout="",
            stderr="",
            exit_code=0,
        )


def test_ruff_scope_excludes_jcodemunch(monkeypatch, tmp_path):
    backend = FakeBackend()
    step = RuffFixStep(backend=backend)
    repo_root = tmp_path
    step.run(repo_root)
    assert backend.last_cmd[:5] == ["uv", "run", "ruff", "check", "--fix"]
    assert backend.last_cmd[5:] == ["harness/policy.py"]
    assert "jcodemunch-mcp" not in backend.last_cmd


def test_basedpyright_scope_excludes_jcodemunch(monkeypatch, tmp_path):
    backend = FakeBackend()
    step = BasedPyrightStep(backend=backend)
    repo_root = tmp_path
    step.run(repo_root)
    assert backend.last_cmd[:3] == ["uv", "run", "basedpyright"]
    assert backend.last_cmd[3:] == ["harness/policy.py"]
    assert "jcodemunch-mcp" not in backend.last_cmd
