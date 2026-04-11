"""Backward-compatibility shim — use harness.telemetry instead."""
# COMPAT SHIM: Re-exports from harness.telemetry.
# All new code should import from harness.telemetry directly.

from harness.telemetry import *  # noqa: F401,F403
from harness.telemetry import __getattr__ as _telemetry_getattr  # noqa: F401

from typing import Any


def __getattr__(name: str) -> Any:
    return _telemetry_getattr(name)
