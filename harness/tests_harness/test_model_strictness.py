from __future__ import annotations

import pytest

from harness.agents.common import UsageEnvelope
from harness.orchestration.types import EvaluationMetrics
from harness.prompts import WriterOptimizationBrief


def test_evaluation_metrics_rejects_unknown_fields():
    with pytest.raises(Exception):
        EvaluationMetrics.model_validate({"score": 1.0, "unexpected": 2})  # type: ignore[arg-type]


def test_writer_optimization_brief_rejects_unknown_fields():
    with pytest.raises(Exception):
        WriterOptimizationBrief.model_validate({"findings": [], "extra": "nope"})  # type: ignore[arg-type]


def test_usage_envelope_flattening_and_rejection():
    env = UsageEnvelope.model_validate(
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "prompt_details": [{"name": "cached", "value": 4}]}
    )
    flat = env.model_dump_flat()
    assert flat["prompt_tokens"] == 1
    assert flat["completion_tokens"] == 2
    assert flat["total_tokens"] == 3
    assert flat["prompt_tokens_details.cached"] == 4
    with pytest.raises(Exception):
        UsageEnvelope.model_validate({"prompt_tokens": 1, "extra": {"unexpected": True}})  # type: ignore[arg-type]
