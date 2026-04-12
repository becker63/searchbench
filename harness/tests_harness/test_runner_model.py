from __future__ import annotations

import os

from harness.agents.localizer import resolve_runner_model


def test_runner_model_defaults_to_gpt_oss(monkeypatch):
    monkeypatch.delenv("RUNNER_MODEL", raising=False)
    resolved = resolve_runner_model()
    assert resolved == "gpt-oss-120b"


def test_runner_model_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("RUNNER_MODEL", "llama3.1-8b")
    resolved = resolve_runner_model("custom-model")
    assert resolved == "custom-model"
