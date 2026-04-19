from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class StepResult(BaseModel):
    """
    Canonical pipeline step result used across pipeline and loop.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str
    success: bool
    stdout: str
    stderr: str
    exit_code: int

    @classmethod
    def from_external(cls, step: object) -> "StepResult":
        """Validate and coerce external step payloads."""
        if isinstance(step, BaseModel):
            return cls.model_validate(step.model_dump())
        if isinstance(step, dict):
            return cls.model_validate(step)
        data: dict[str, Any] = {}
        allowed_fields = {"name", "success", "stdout", "stderr", "exit_code"}
        extras: set[str] = set()
        for field in allowed_fields:
            if hasattr(step, field):
                data[field] = getattr(step, field)
        if hasattr(step, "__dict__"):
            extras = set(getattr(step, "__dict__", {}).keys()) - allowed_fields
        if extras:
            raise ValueError(f"Unexpected fields in step payload: {sorted(extras)}")
        return cls.model_validate(data)

    def to_mapping(self) -> dict[str, object]:
        return self.model_dump()


class PipelineClassification(BaseModel):
    """Typed classification of pipeline outcomes."""

    model_config = ConfigDict(extra="forbid")

    type_errors: str = ""
    lint_errors: str = ""
    test_failures: str = ""
    passed: list[str] = Field(default_factory=list)

class Step(Protocol):
    name: str

    def run(self, repo_root: Path) -> StepResult:
        ...


class ExecutionBackend:
    def run(self, cmd: list[str], cwd: Path, name: str | None = None) -> StepResult:
        import subprocess
        from ..utils.repo_root import find_repo_root

        allowed_prefixes = {
            ("uv", "run", "ruff", "check", "--fix"),
            ("uv", "run", "basedpyright"),
            ("uv", "run", "pytest", "iterative-context"),
            ("uv", "run", "pytest", "iterative-context", "-q", "--disable-warnings", "--maxfail=1"),
        }

        repo_root = find_repo_root()
        if cwd != repo_root:
            return StepResult(
                name=" ".join(cmd),
                success=False,
                stdout="",
                stderr="Execution not allowed outside repo root",
                exit_code=1,
            )

        def _is_allowed(command: list[str]) -> bool:
            for prefix in allowed_prefixes:
                if tuple(command[: len(prefix)]) == prefix:
                    return True
            return False

        if not _is_allowed(cmd):
            return StepResult(
                name=" ".join(cmd),
                success=False,
                stdout="",
                stderr="Command not permitted",
                exit_code=1,
            )

        completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        result = StepResult(
            name=" ".join(cmd),
            success=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )
        if name:
            result.name = name
        return result
