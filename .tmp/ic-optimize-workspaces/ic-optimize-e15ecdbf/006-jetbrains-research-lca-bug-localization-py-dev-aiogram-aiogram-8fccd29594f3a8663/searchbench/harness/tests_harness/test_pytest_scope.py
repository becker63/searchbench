from pathlib import Path

from harness.pipeline.steps import PytestStep
from harness.pipeline.types import StepResult, ExecutionBackend


class FakeBackend(ExecutionBackend):
    def __init__(self):
        self.last_cmd: list[str] | None = None

    def run(self, cmd: list[str], cwd: Path, name: str | None = None) -> StepResult:
        self.last_cmd = cmd
        return StepResult(
            name=name or "pytest",
            success=True,
            stdout="",
            stderr="",
            exit_code=0,
        )


def test_pytest_scope_limited():
    backend = FakeBackend()
    step = PytestStep(backend=backend)
    repo_root = Path(__file__).resolve().parents[2]
    step.run(repo_root)
    assert backend.last_cmd == ["uv", "run", "pytest", "iterative-context", "-q", "--disable-warnings", "--maxfail=1"]
