#!/usr/bin/env python3
"""Verify that every (see #NNN) / Supersedes #NNN / Superseded by #NNN
reference in DECISIONS.md resolves to a real ## NNN. entry in the same file.

Content inside fenced code blocks is skipped — format examples intentionally
use illustrative numbers that are not guaranteed to match real entries.

Usage:
    .venv/bin/python scripts/check_decision_refs.py
    .venv/bin/python scripts/check_decision_refs.py --file path/to/DECISIONS.md

Exit 0 if all references resolve; exit 1 on any dangling reference or
duplicate entry number.

Limitation: This checker verifies that every #NNN reference resolves to SOME
## NNN. entry. It does not verify that the reference points to the CORRECT
entry. A reference written as (see #007) when the author meant (see #009)
will pass this check if both #007 and #009 exist as entries. Semantic review
still requires a human.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DECISIONS_PATH = Path("DECISIONS.md")

# Matches the header line that defines a decision entry.
# Captures the zero-padded three-digit number.
HEADER_RE = re.compile(r"^## (\d{3})\.", re.MULTILINE)

# Matches any #NNN reference in prose. Captures the three-digit number.
# Patterns covered:
#   (see #007)
#   Supersedes #003
#   Superseded by #007
#   cited by #002
# Intentionally broad — any #NNN in non-code prose is a candidate.
REF_RE = re.compile(r"#(\d{3})\b")

CODE_FENCE_RE = re.compile(r"^```")


def non_code_spans(text: str) -> list[tuple[int, int]]:
    """Return byte-offset spans of text that are NOT inside fenced code blocks.

    Identical logic to extract_invariants.iter_non_code_spans — see
    DECISIONS.md#004 for why code blocks must be skipped.
    """
    spans: list[tuple[int, int]] = []
    in_fence = False
    cursor = 0

    for m in re.finditer(r"^.*$", text, re.MULTILINE):
        line = m.group(0)
        line_start = m.start()
        line_end = m.end()

        if CODE_FENCE_RE.match(line):
            if not in_fence:
                if line_start > cursor:
                    spans.append((cursor, line_start))
                in_fence = True
            else:
                in_fence = False
                cursor = line_end + 1

    if not in_fence and cursor < len(text):
        spans.append((cursor, len(text)))

    return spans


def check(path: Path) -> int:
    """Check cross-references in *path*. Returns exit code (0 = ok, 1 = fail)."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    # Collect defined entry numbers. Headers, like references, must come from
    # non-code-block prose — the format-example in the Supersession section
    # contains an illustrative "## 007. Celery..." header inside a fenced
    # code block that would otherwise register as a duplicate of the real #007.
    # Track counts rather than a set so duplicate ## NNN. headers are caught
    # rather than silently deduped. Two entries with the same number would
    # render in GitHub but break the "numbers are stable anchors" guarantee
    # the format depends on.
    header_counts: dict[str, int] = {}
    for span_start, span_end in non_code_spans(text):
        chunk = text[span_start:span_end]
        for m in HEADER_RE.finditer(chunk):
            header_counts[m.group(1)] = header_counts.get(m.group(1), 0) + 1

    duplicates = sorted(n for n, c in header_counts.items() if c > 1)
    if duplicates:
        print(
            f"error: {path} has duplicate ## NNN. header(s): "
            f"{', '.join(f'#{n} ({header_counts[n]}x)' for n in duplicates)}",
            file=sys.stderr,
        )
        print(
            "Fix: entry numbers must be unique. Renumber the duplicates — "
            "remember that numbers are never reused (see DECISIONS.md header).",
            file=sys.stderr,
        )
        return 1

    defined = set(header_counts)

    # Collect references only from non-code prose.
    refs: list[tuple[str, int]] = []  # (number, offset)
    for span_start, span_end in non_code_spans(text):
        chunk = text[span_start:span_end]
        for m in REF_RE.finditer(chunk):
            refs.append((m.group(1), span_start + m.start()))

    # Diff.
    errors: list[str] = []
    for num, offset in refs:
        if num not in defined:
            # Find approximate line number for a useful error message.
            line_no = text[:offset].count("\n") + 1
            errors.append(f"  line {line_no}: #{num} is referenced but no ## {num}. entry exists")

    if errors:
        print(f"error: {path} has {len(errors)} unresolved cross-reference(s):", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        print(
            "\nFix: add the missing ## NNN. entry or correct the reference number.",
            file=sys.stderr,
        )
        return 1

    print(
        f"ok: {path} — {len(defined)} entries defined, "
        f"{len(refs)} cross-references checked, all resolve."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--file",
        type=Path,
        default=DECISIONS_PATH,
        help=f"Path to DECISIONS.md (default: {DECISIONS_PATH})",
    )
    args = parser.parse_args()
    return check(args.file)


if __name__ == "__main__":
    sys.exit(main())
