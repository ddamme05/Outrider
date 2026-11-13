"""Keep BASE/HEAD provenance exact when PR edits shift a candidate's lines."""

from __future__ import annotations

from difflib import unified_diff

import pytest

from outrider.coordinates import candidate_change_context


def _patch(base: str, head: str, *, old_path: str = "app.py", new_path: str = "app.py") -> str:
    return "".join(
        unified_diff(
            base.splitlines(keepends=True),
            head.splitlines(keepends=True),
            fromfile="a/" + old_path,
            tofile="b/" + new_path,
        )
    )


def _context(base: str, head: str, start: int, end: int, **changes):
    arguments = {
        "file_path": "app.py",
        "previous_path": None,
        "status": "modified",
        "patch": _patch(base, head),
        "base_source": base,
        "head_source": head,
        "line_start": start,
        "line_end": end,
        "max_diff_bytes": 2000,
        **changes,
    }
    return candidate_change_context(**arguments)


def test_shifted_unchanged_claim_maps_to_original_base_lines():
    base = "".join(f"value_{n} = {n}\n" for n in range(1, 25))
    head = "# inserted\n" + base
    result = _context(base, head, 18, 20)
    assert result.availability == "available"
    assert result.base_ranges == ((17, 19),)
    assert result.head_lines_added is False
    assert result.base_lines_removed is False
    assert result.diff == ""


@pytest.mark.parametrize(
    ("base", "head", "start", "ranges", "added", "removed", "old_line"),
    [
        ("if authorized:\n    allow()\n", "allow()\n", 1, ((1, 2),), True, True, "-if authorized:"),
        ("check()\nrun()\n", "run()\n", 1, ((1, 2),), False, True, "-check()"),
    ],
)
def test_replaced_and_deleted_guards_retain_base_evidence(
    base, head, start, ranges, added, removed, old_line
):
    result = _context(base, head, start, start)
    assert result.availability == "available"
    assert result.base_ranges == ranges
    assert result.head_lines_added is added
    assert result.base_lines_removed is removed
    assert old_line in result.diff


def test_added_file_has_no_invented_base_counterpart():
    head = "SECRET = 'sample'\n"
    result = _context("", head, 1, 1, status="added", base_source=None)
    assert result.availability == "added_file"
    assert result.base_path is None and result.base_ranges == ()
    assert result.head_lines_added is True
    assert "+SECRET" in result.diff


def test_renamed_file_uses_original_base_path():
    base, head = "keep()\n", "start()\nkeep()\n"
    result = _context(
        base,
        head,
        2,
        2,
        file_path="new.py",
        previous_path="old.py",
        status="renamed",
        patch=_patch(base, head, old_path="old.py", new_path="new.py"),
    )
    assert result.availability == "available"
    assert result.head_path == "new.py" and result.base_path == "old.py"
    assert result.base_ranges == ((1, 1),)


@pytest.mark.parametrize(
    ("patch", "availability"),
    [
        (None, "patch_unavailable"),
        ("", "invalid_patch"),
        ("@@ -1,2 +1,2 @@\n-old()\n+new()\n", "invalid_patch"),
        ("@@ -1 +1 @@\n-old()\n+wrong()\n", "source_mismatch"),
    ],
)
def test_unavailable_patch_never_guesses_same_numbered_base_lines(patch, availability):
    result = _context("old()\n", "new()\n", 1, 1, patch=patch)
    assert result.availability == availability
    assert result.base_ranges == ()
    assert result.head_lines_added is None


def test_partial_patch_with_hidden_changes_is_not_trusted():
    base = "".join(f"line_{n}\n" for n in range(30))
    head = base.replace("line_0\n", "changed_0\n").replace("line_29\n", "changed_29\n")
    last_change_only = "@@ -28,3 +28,3 @@\n line_27\n line_28\n-line_29\n+changed_29\n"
    result = _context(base, head, 15, 15, patch=last_change_only)
    assert result.availability == "source_mismatch"
    assert result.base_ranges == ()


def test_diff_cap_counts_utf8_bytes_and_keeps_complete_lines():
    base, head = "old()\n", "snowman = '☃☃☃'\n"
    result = _context(base, head, 1, 1, max_diff_bytes=33)
    assert result.availability == "available"
    assert len(result.diff.encode()) <= 33
    assert result.diff_truncated
    assert result.diff.endswith("\n")
    assert "snowman" not in result.diff


def test_explicit_empty_patch_with_identical_captures_retains_identity_mapping():
    source = "keep()\n"
    result = _context(source, source, 1, 1, patch="")
    assert result.availability == "available"
    assert result.base_ranges == ((1, 1),)
    assert result.head_lines_added is False and result.base_lines_removed is False
