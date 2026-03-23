import sys
from pathlib import Path


def _bootstrap_repo_root() -> Path:
    """
    Locate the monorepo root before importing harness utilities.
    """
    start = Path(__file__).resolve()
    for path in [start, *start.parents]:
        if path.name == "searchbench":
            return path
        if (path / "pyproject.toml").exists() and (path / "uv.lock").exists():
            return path
    raise FileNotFoundError("Repository root not found for tests")


ROOT = _bootstrap_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import after sys.path is set to reuse the shared helper elsewhere.
from harness.utils.repo_root import find_repo_root  # noqa: E402
