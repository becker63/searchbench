# pyright: reportMissingTypeStubs=false
import os

import pytest
from hypothesis import settings
from pytest_snapshot.plugin import Snapshot

from iterative_context.test_helpers import GraphSnapshot

# Hypothesis shrinking is enabled by default.
# Keep tests simple and deterministic so shrinking produces minimal counterexamples.

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
)

settings.register_profile(
    "debug",
    max_examples=10,
    deadline=None,
    print_blob=True,
)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture
def snapshot_graph(snapshot: Snapshot) -> GraphSnapshot:
    """Provide a GraphSnapshot instance bound to the pytest‑snapshot fixture."""
    return GraphSnapshot(snapshot)
