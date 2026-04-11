"""Backend dispatch: IC, JCodeMunch, and MCP adapters."""

from .ic import IterativeContextBackend
from .jc import JCodeMunchBackend

__all__ = ["IterativeContextBackend", "JCodeMunchBackend"]
