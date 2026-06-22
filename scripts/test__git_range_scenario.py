#!/usr/bin/env python3
"""Smoke tests for _git_range_scenario.py — the wire-fidelity-critical pure logic.

Run: `uv run python scripts/test__git_range_scenario.py`

These pin the parts a faithful GitHub `/pulls/{n}/files` reconstruction depends on,
none of which need git (they take synthetic patch text):
  (a) _hunks_only strips `diff --git` / `index` / `--- a/` / `+++ b/` headers,
      leaving a body that starts with `@@`
  (b) _hunks_only returns None for a no-hunk (pure-rename / mode-only) patch and
      for an empty string — matching GitHub omitting `patch` for those
  (c) _count_changes derives additions/deletions from the unified diff the way
      GitHub does (and does NOT miscount the `+++`/`---` header lines)
  (d) _parse_range accepts two-dot `A..B` (incl. refs with single dots like
      v1.2.3) and rejects three-dot `A...B`, shell-noise, and malformed input
  (e) FileEntry bounds: changed_bytes < head_bytes, floor <= ceiling, is_python
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _git_range_scenario import (  # noqa: E402 — sys.path set above
    FileEntry,
    GitRangeError,
    _count_changes,
    _hunks_only,
    _parse_range,
)

_failures: list[str] = []


def _check(name: str, cond: bool) -> None:  # noqa: FBT001 — tiny local assert helper
    mark = "ok" if cond else "FAIL"
    print(f"  [{mark}] {name}")
    if not cond:
        _failures.append(name)


# A realistic `git diff` chunk for a modified file: headers THEN hunks.
_GIT_DIFF_MODIFIED = (
    "diff --git a/src/x.py b/src/x.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/src/x.py\n"
    "+++ b/src/x.py\n"
    "@@ -1,3 +1,4 @@\n"
    " import os\n"
    "-x = 1\n"
    "+x = 2\n"
    "+y = 3\n"
    " print(x)\n"
)

# A pure rename with NO content change: git emits similarity/rename lines, no @@.
_GIT_DIFF_PURE_RENAME = (
    "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n"
)


def test_hunks_only_strips_headers() -> None:
    body = _hunks_only(_GIT_DIFF_MODIFIED) or ""
    _check("hunks_only: returns a body", body != "")
    _check("hunks_only: body starts with @@", body.startswith("@@"))
    _check("hunks_only: no 'diff --git' header", "diff --git" not in body)
    _check("hunks_only: no '--- a/' header", "--- a/" not in body)
    _check("hunks_only: no '+++ b/' header", "+++ b/" not in body)
    _check("hunks_only: keeps the added content line", "+x = 2" in body)


def test_hunks_only_none_cases() -> None:
    _check("hunks_only: pure rename -> None", _hunks_only(_GIT_DIFF_PURE_RENAME) is None)
    _check("hunks_only: empty string -> None", _hunks_only("") is None)


def test_count_changes_matches_github() -> None:
    body = _hunks_only(_GIT_DIFF_MODIFIED)
    additions, deletions = _count_changes(body)
    # 2 added (+x = 2, +y = 3), 1 removed (-x = 1); the +++/--- headers are gone
    # already, but the counter must ignore them even if present.
    _check("count_changes: 2 additions", additions == 2)
    _check("count_changes: 1 deletion", deletions == 1)
    _check("count_changes: None patch -> (0, 0)", _count_changes(None) == (0, 0))
    # Counter must not miscount the file-header lines as +/- content.
    add2, del2 = _count_changes("--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n")
    _check("count_changes: ignores +++/--- headers", (add2, del2) == (1, 1))


def test_parse_range() -> None:
    _check("parse_range: A..B", _parse_range("0c70d18^..39c538b") == ("0c70d18^", "39c538b"))
    # Refs with single dots must survive (the separator is the run of TWO dots).
    _check("parse_range: dotted refs", _parse_range("v1.2.3..v1.2.4") == ("v1.2.3", "v1.2.4"))
    # Reflog/peel refs are valid endpoints.
    _check("parse_range: reflog ref", _parse_range("HEAD@{1}..HEAD") == ("HEAD@{1}", "HEAD"))
    # Rejected: three-dot, embedded '..', leading/trailing dot, empty side, shell noise.
    for bad in (
        "main...HEAD",  # three-dot
        "main..HEAD..other",  # embedded '..' (3 parts)
        ".main..HEAD",  # leading dot in START
        "main..HEAD.",  # trailing dot in END
        "a..",  # empty END
        "..b",  # empty START
        "garbage",  # no separator
        "A..B; rm -rf /",  # shell metacharacters
        "A B",  # space, no separator
        "",  # empty
    ):
        rejected = False
        try:
            _parse_range(bad)
        except GitRangeError:
            rejected = True
        _check(f"parse_range: rejects {bad!r}", rejected)


def test_file_entry_bounds() -> None:
    entry = FileEntry(
        path="src/x.py",
        status="modified",
        additions=2,
        deletions=1,
        patch="@@ -1,3 +1,4 @@\n import os\n-x = 1\n+x = 2\n+y = 3\n print(x)\n",
        content_base="import os\nx = 1\nprint(x)\n",
        content_head="import os\nx = 2\ny = 3\nprint(x)\n" + ("# pad\n" * 50),
        previous_path=None,
    )
    _check("FileEntry: is_python", entry.is_python is True)
    _check(
        "FileEntry: changed_bytes < head_bytes (big file, small diff)",
        entry.changed_bytes < entry.head_bytes,
    )
    _check(
        "FileEntry: floor <= ceiling",
        entry.estimated_tokens_floor <= entry.estimated_tokens_ceiling,
    )
    nonpy = FileEntry("d/openapi.json", "modified", 1, 0, "@@ -1 +1 @@\n+{}\n", "{}", "{}", None)
    _check("FileEntry: .json is not python", nonpy.is_python is False)


def main() -> int:
    for test in (
        test_hunks_only_strips_headers,
        test_hunks_only_none_cases,
        test_count_changes_matches_github,
        test_parse_range,
        test_file_entry_bounds,
    ):
        test()
    print()
    if _failures:
        print(f"  FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("  _git_range_scenario smoke tests pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
