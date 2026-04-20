from __future__ import annotations

from enum import Enum
from typing import Any


class LocalizationFailureCategory(str, Enum):
    MATERIALIZATION = "materialization"
    RUNNER = "runner"
    SCORING = "scoring"
    UNKNOWN = "unknown"


class LocalizationEvaluationError(Exception):
    """
    Typed localization evaluation failure.
    """

    def __init__(
        self,
        category: LocalizationFailureCategory,
        message: str,
        task_id: str | None = None,
        observability_summary: Any | None = None,
    ):
        super().__init__(message)
        self.category = category
        self.task_id = task_id
        self.observability_summary = observability_summary
