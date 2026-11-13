"""Whole-file deletion evidence must reproduce the captured BASE exactly."""

import pytest

from outrider.coordinates import removed_source_to_review_body, validate_removed_file
from outrider.coordinates.errors import CoordinateError


@pytest.mark.parametrize(
    ("base", "patch"),
    [
        ("require_auth(user)\n", "@@ -1 +0,0 @@\n-require_auth(user)\n"),
        ("é = 1", "@@ -1 +0,0 @@\n-é = 1\n\\ No newline at end of file\n"),
    ],
)
def test_complete_removal_maps_only_to_review_body(base, patch):
    kwargs = {
        "file_path": "src/auth.py",
        "status": "removed",
        "base_content": base,
        "head_content": None,
        "patch": patch,
    }
    assert sum(hunk.source_length for hunk in validate_removed_file(**kwargs)) == 1
    assert removed_source_to_review_body(**kwargs, line_start=1, line_end=1) == "src/auth.py"
    for start, end in ((2, 2), (1, 2)):
        with pytest.raises(CoordinateError):
            removed_source_to_review_body(**kwargs, line_start=start, line_end=end)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "modified"},
        {"status": "renamed"},
        {"head_content": ""},
        {"base_content": None},
        {"base_content": "require_auth(other)\n"},
        {"base_content": "require_auth(user)\nrun()\n"},
        {"patch": None},
        {"patch": "@@ -1 +1 @@\n-require_auth(user)\n+run()\n"},
        {"patch": "--- a/other.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-require_auth(user)\n"},
        {"file_path": "../auth.py"},
    ],
)
def test_partial_or_unbound_removed_source_is_refused(change):
    kwargs = {
        "file_path": "src/auth.py",
        "status": "removed",
        "base_content": "require_auth(user)\n",
        "head_content": None,
        "patch": "@@ -1 +0,0 @@\n-require_auth(user)\n",
        **change,
    }
    with pytest.raises(CoordinateError):
        validate_removed_file(**kwargs)
