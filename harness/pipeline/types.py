from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Any


@dataclass
class StepResult:
    """
    Canonical pipeline step result used across pipeline and loop.
    """

    name: str
    success: bool
    stdout: str
    stderr: str
    exit_code: int

    @classmethod
    def from_mapping(cls, step: Mapping[str, Any] | Any) -> "StepResult":
        if isinstance(step, Mapping):
            name = str(step.get("name") or "")
            success = bool(step.get("success", False))
            exit_code_val = step.get("exit_code")
            exit_code = int(exit_code_val) if isinstance(exit_code_val, int) else 0
            stdout = str(step.get("stdout") or "")
            stderr = str(step.get("stderr") or "")
            return cls(name=name, success=success, stdout=stdout, stderr=stderr, exit_code=exit_code)
        name = getattr(step, "name", "") or ""
        success = bool(getattr(step, "success", False))
        exit_code_val = getattr(step, "exit_code", 0)
        exit_code = int(exit_code_val) if isinstance(exit_code_val, int) else 0
        stdout = str(getattr(step, "stdout", "") or "")
        stderr = str(getattr(step, "stderr", "") or "")
        return cls(name=name, success=success, stdout=stdout, stderr=stderr, exit_code=exit_code)

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


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
