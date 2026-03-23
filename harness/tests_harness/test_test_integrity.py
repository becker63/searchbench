from pathlib import Path

from harness.loop import _hash_tests_dir, _write_policy, _read_policy


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
        policy_path = repo_root / "harness" / "policy.py"
        _write_policy(_read_policy(policy_path), policy_path)
