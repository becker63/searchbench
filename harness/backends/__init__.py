"""Concrete backend implementations over the shared MCP utility layer."""

from .ic import IterativeContextBackend
from .jc import JCodeMunchBackend

__all__ = ["IterativeContextBackend", "JCodeMunchBackend"]
