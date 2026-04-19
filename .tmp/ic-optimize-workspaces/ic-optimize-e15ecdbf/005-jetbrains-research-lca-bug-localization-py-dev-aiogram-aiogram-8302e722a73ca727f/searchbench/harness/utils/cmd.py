from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline.types import StepResult


def run_cmd(cmd: List[str], cwd: Path) -> "StepResult":
    completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    from ..pipeline.types import StepResult

    return StepResult(
        name=" ".join(cmd),
        success=completed.returncode == 0,
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
    )
