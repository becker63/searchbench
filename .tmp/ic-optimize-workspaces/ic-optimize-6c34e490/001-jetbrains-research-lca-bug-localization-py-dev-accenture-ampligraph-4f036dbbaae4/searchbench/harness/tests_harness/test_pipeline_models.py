from __future__ import annotations

import pytest

from harness.pipeline.types import StepResult


def test_step_result_from_external_rejects_unknown_fields():
    class External:
        def __init__(self):
            self.name = "step"
            self.success = True
            self.stdout = ""
            self.stderr = ""
            self.exit_code = 0
            self.extra = "not-allowed"

    with pytest.raises(ValueError):
        StepResult.from_external(External())

    with pytest.raises(Exception):
        StepResult.model_validate({"name": "s", "success": True, "stdout": "", "stderr": "", "exit_code": 0, "extra": "x"})
