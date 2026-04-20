from __future__ import annotations

from pathlib import Path

from .types import ExecutionBackend, StepResult


# Only validate the generated policy surface during optimization.
_TARGET_PATHS = ["harness/policy/current.py"]


class _BaseStep:
    name: str = ""

    def __init__(self, backend: ExecutionBackend | None = None) -> None:
        self.backend: ExecutionBackend = backend or ExecutionBackend()


class RuffFixStep(_BaseStep):
    name = "ruff_fix"

    def run(self, repo_root: Path) -> StepResult:
        return self.backend.run(
            ["uv", "run", "ruff", "check", "--fix", *_TARGET_PATHS],
            cwd=repo_root,
            name=self.name,
        )


class BasedPyrightStep(_BaseStep):
    name = "basedpyright"

    def run(self, repo_root: Path) -> StepResult:
        return self.backend.run(
            ["uv", "run", "basedpyright", *_TARGET_PATHS],
            cwd=repo_root,
            name=self.name,
        )


class PytestStep(_BaseStep):
    name = "pytest_iterative_context"

    def run(self, repo_root: Path) -> StepResult:
        return self.backend.run(
            ["uv", "run", "pytest", "iterative-context", "-q", "--disable-warnings", "--maxfail=1"],
            cwd=repo_root,
            name=self.name,
        )
