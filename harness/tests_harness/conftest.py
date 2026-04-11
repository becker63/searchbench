from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from pydantic import BaseModel, ConfigDict, Field
from harness import observability


class TestSpan(BaseModel):
    """Pydantic test span with scoring support for Langfuse stubs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    trace_id: str | None = None
    scores: list[tuple[str, float, dict[str, object] | None]] = Field(default_factory=list)

    def score(self, name: str, value: float | int | bool, metadata: dict[str, object] | None = None) -> None:
        self.scores.append((name, float(value), metadata))

    def end(self, **kwargs: Any) -> None:  # pragma: no cover - simple stub
        return None


@pytest.fixture
def test_span() -> TestSpan:
    """Reusable span fixture with scoring and trace id."""
    return TestSpan(id="span", trace_id="trace")


@pytest.fixture(autouse=True)
def _clear_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Langfuse creds are absent to prevent accidental network calls."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_ENV", raising=False)


@pytest.fixture(autouse=True)
def _stub_langfuse_client(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Provide a dummy Langfuse client to prevent network access in tests."""
    import os
    if os.getenv("LANGFUSE_NO_STUB") == "1":
        return
    if request.node.get_closest_marker("no_langfuse_stub"):
        return
    import harness.telemetry.langfuse as lf
    import harness.telemetry.score_emitter as se

    monkeypatch.setattr(lf, "_STRICT_DEBUG", False)
    monkeypatch.setattr(se, "_STRICT_DEBUG", False)

    class DummyTrace:
        def __init__(self, session_id: str | None = None):
            self.id = "trace"
            self.session_id = session_id

        def span(self, name: str, metadata=None, input=None, session_id=None):
            return DummySpan(session_id=session_id or self.session_id)

    class DummySpan(DummyTrace):
        def __init__(self, session_id: str | None = None):
            super().__init__(session_id=session_id)
            self.trace_id = "trace"

        def end(self, **kwargs: Any) -> None:
            return None

        def update(self, **kwargs: Any) -> None:
            return None

    class DummyClient:
        def __init__(self):
            self.scores: list[dict[str, object]] = []

        def create_score(self, **kwargs: Any) -> None:
            self.scores.append(kwargs)

        def trace(self, **kwargs: Any) -> DummyTrace:
            return DummyTrace(session_id=kwargs.get("session_id"))

        @contextmanager
        def start_as_current_observation(self, **kwargs: Any) -> Iterator[DummySpan]:
            yield DummySpan()

        def auth_check(self) -> bool:
            return True

        def flush(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    original = getattr(lf, "get_langfuse_client", None)
    monkeypatch.setattr(lf, "_real_get_langfuse_client", original, raising=False)
    monkeypatch.setattr(lf, "get_langfuse_client", lambda: DummyClient())


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "no_langfuse_stub: disable Langfuse stub fixture")
