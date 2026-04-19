from __future__ import annotations

import warnings

from harness.utils.warnings import suppress_third_party_invalid_escape_warnings


def _emit_invalid_escape_warning(filename: str = "/tmp/target_repo/file.py") -> None:
    compile('pattern = "\\/"\n', filename, "exec")


def test_invalid_escape_warning_is_suppressed_only_inside_target_repo_boundary():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with suppress_third_party_invalid_escape_warnings("/tmp/target_repo"):
            _emit_invalid_escape_warning()

    assert captured == []


def test_invalid_escape_warning_still_surfaces_outside_target_repo_boundary():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _emit_invalid_escape_warning("/tmp/harness/file.py")

    assert any(
        item.category is SyntaxWarning and "invalid escape sequence" in str(item.message)
        for item in captured
    )


def test_other_harness_warnings_are_not_suppressed_inside_boundary():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with suppress_third_party_invalid_escape_warnings("/tmp/target_repo"):
            warnings.warn("harness warning", RuntimeWarning, stacklevel=1)

    assert any(
        item.category is RuntimeWarning and str(item.message) == "harness warning"
        for item in captured
    )

