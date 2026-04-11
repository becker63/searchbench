"""Backward-compatibility shim for tools package."""
# COMPAT SHIM: backends moved to harness.backends, mcp_adapter to harness.backends.mcp.
from __future__ import annotations

import harness.backends as backends  # noqa: F401

__all__ = ["backends"]
