from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import warnings


@contextmanager
def suppress_third_party_invalid_escape_warnings(repo_root: str | Path | None = None) -> Iterator[None]:
    """Suppress noisy invalid-escape SyntaxWarnings while analyzing target repos.

    Python emits these warnings when code is compiled or parsed, even when the
    benchmark repository is only being inspected. Keep the filter scoped to
    target-repo analysis boundaries so harness warnings still surface normally.
    """
    del repo_root
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"invalid escape sequence .*",
            category=SyntaxWarning,
        )
        yield

