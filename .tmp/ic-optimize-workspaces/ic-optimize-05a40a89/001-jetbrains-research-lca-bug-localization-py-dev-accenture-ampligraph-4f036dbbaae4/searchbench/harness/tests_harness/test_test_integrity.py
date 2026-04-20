import hashlib
from pathlib import Path

from harness.orchestration.evaluation import _read_policy
from harness.orchestration.writer import _write_policy


def _hash_tests_dir(repo_root: Path) -> str:
    digest = hashlib.sha256()
    tests_dir = repo_root / "iterative-context" / "tests"
    for path in sorted(p for p in tests_dir.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(tests_dir)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_test_directory_integrity(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    tests_dir = repo_root / "iterative-context" / "tests"
    marker = tests_dir / "tmp_integrity_marker.py"
    before = _hash_tests_dir(repo_root)
    try:
        marker.write_text("# marker\n", encoding="utf-8")
        after = _hash_tests_dir(repo_root)
        assert before != after
    finally:
        if marker.exists():
            marker.unlink()
        # ensure policy restored to avoid side effects
        policy_path = repo_root / "harness" / "policy" / "current.py"
        _write_policy(_read_policy(policy_path), policy_path)
