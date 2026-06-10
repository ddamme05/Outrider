#!/usr/bin/env python3
"""Extract tagged invariants from docs/spec.md into docs/invariants.md.

Tag format (inline in spec.md):

    <!-- invariant:id=<kebab-case-id>
         rule: <one-sentence rule statement>
         violation: <one-sentence description of violation patterns>
         check: <optional: shell command that verifies the invariant>
    -->

Rules enforced at extraction time (all are hard failures, not warnings):
- The 'id' is unique across the entire spec.
- No tag field contains a section number (e.g. '§7.4' or '7.4').
  Section references are derived from the enclosing heading at extraction
  time, so embedding one in the tag creates drift.
- Every tag sits inside a numbered H2/H3/H4 heading's scope. Tags in the
  preamble (before any heading) fail.
- Tags inside fenced code blocks are silently skipped (they're example
  documentation, not live invariants).

The generated invariants.md lists entries alphabetically by ID. The
[security-critical] label is applied by presence of 'security:' field
(see tag schema below).

Usage:
    .venv/bin/python scripts/extract_invariants.py
    .venv/bin/python scripts/extract_invariants.py --check   # verify only; no write
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

SPEC_PATH = Path("docs/spec.md")
OUT_PATH = Path("docs/invariants.md")

# The tag regex is intentionally forgiving about whitespace inside the
# block, but strict about structure. The trailing '-->' must appear.
TAG_RE = re.compile(
    r"<!--\s*invariant:id=(?P<id>[a-z0-9][a-z0-9-]*)\s*\n"
    r"(?P<body>.*?)"
    r"-->",
    re.DOTALL,
)

FIELD_RE = re.compile(
    r"^\s*(?P<key>rule|violation|check|security):\s*(?P<val>.+?)\s*$",
    re.MULTILINE,
)

# Detect hardcoded section references. Matches explicit citations only:
#   '§7.4', '§ 7.4', 'section 7.4', 'section 7.4.1', 'see §7'.
# Does NOT match decimal values (0.9, 1.5), version strings (Python 3.13),
# or other legitimate numeric content. The signal we care about is the
# intent to cite a spec section, marked either by the § glyph or the
# word 'section'.
SECTION_RE = re.compile(
    r"§\s*\d+(?:\.\d+)*"
    r"|\bsection\s+\d+(?:\.\d+)+\b",
    re.IGNORECASE,
)

HEADING_RE = re.compile(r"^(#{2,4})\s+(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<title>.+?)\s*$")
# The optional trailing '.' after the number accommodates the spec's
# convention: top-level sections render as '## 1. Title' while subsections
# render as '### 1.1 Title'.

CODE_FENCE_RE = re.compile(r"^```")


class ExtractorError(Exception):
    """Raised for any validation failure. Halts extraction."""


@dataclass
class Invariant:
    id: str
    rule: str
    violation: str
    check: str | None
    security_critical: bool
    section_num: str
    section_title: str
    source_offset: int  # byte offset in spec.md, for error messages


def iter_non_code_spans(text: str) -> list[tuple[int, int]]:
    """Return byte spans that are NOT inside fenced code blocks.

    We need this so TAG_RE matches aren't claimed by example tags inside
    ```markdown blocks in the spec itself.
    """
    spans: list[tuple[int, int]] = []
    in_fence = False
    cursor = 0

    for line_match in re.finditer(r"^.*$", text, re.MULTILINE):
        line = line_match.group(0)
        line_start = line_match.start()
        line_end = line_match.end()

        if CODE_FENCE_RE.match(line):
            if not in_fence:
                # Closing off a non-code span up to the start of this fence line
                if line_start > cursor:
                    spans.append((cursor, line_start))
                in_fence = True
            else:
                in_fence = False
                cursor = line_end + 1  # skip past the closing fence line
        # else: normal line, no span boundary change

    if in_fence:
        raise ExtractorError(
            "Unclosed code fence: the document ends inside a ``` block, so every "
            "tag after the last opening fence would be silently dropped from "
            "extraction (the bug this guards). Close the fence — the ``` marker "
            "count must be even."
        )
    if cursor < len(text):
        spans.append((cursor, len(text)))

    return spans


def find_containing_heading(text: str, offset: int) -> tuple[str, str]:
    """Walk backward from offset to find the nearest numbered H2-H4 heading.

    Returns (section_number, title). Raises ExtractorError if the offset
    is before any numbered heading (i.e. the tag is in the preamble).
    """
    preceding = text[:offset]
    for line in reversed(preceding.split("\n")):
        m = HEADING_RE.match(line)
        if m:
            return m.group("num"), m.group("title").strip()
    raise ExtractorError(
        f"Tag at byte offset {offset} is not inside any numbered "
        f"H2/H3/H4 section. Tags in the preamble are not supported; "
        f"move the tag into the section whose rule it states."
    )


def parse_tag(tag_id: str, body: str, offset: int, spec_text: str) -> Invariant:
    """Parse a single tag match into an Invariant. Validates structure."""
    fields = {m.group("key"): m.group("val").strip() for m in FIELD_RE.finditer(body)}

    required = {"rule", "violation"}
    missing = required - fields.keys()
    if missing:
        raise ExtractorError(
            f"Tag {tag_id!r} at byte offset {offset} is missing "
            f"required field(s): {sorted(missing)}"
        )

    # Section-number-in-tag check. Hard fail.
    for field_name in ("rule", "violation", "check"):
        val = fields.get(field_name)
        if val and SECTION_RE.search(val):
            raise ExtractorError(
                f"Tag {tag_id!r} at byte offset {offset} contains a "
                f"hardcoded section reference in its {field_name!r} field. "
                f"Remove it \u2014 Source is derived from the enclosing "
                f"heading automatically. If you need a cross-reference to "
                f"another invariant, reference it by ID, not by section."
            )

    section_num, section_title = find_containing_heading(spec_text, offset)

    return Invariant(
        id=tag_id,
        rule=fields["rule"],
        violation=fields["violation"],
        check=fields.get("check"),
        security_critical="security" in fields and fields["security"].lower() == "critical",
        section_num=section_num,
        section_title=section_title,
        source_offset=offset,
    )


def extract(spec_text: str) -> list[Invariant]:
    """Extract all invariants from spec text. Validates uniqueness."""
    non_code_spans = iter_non_code_spans(spec_text)

    invariants: list[Invariant] = []
    for span_start, span_end in non_code_spans:
        chunk = spec_text[span_start:span_end]
        for match in TAG_RE.finditer(chunk):
            invariants.append(
                parse_tag(
                    tag_id=match.group("id"),
                    body=match.group("body"),
                    offset=span_start + match.start(),
                    spec_text=spec_text,
                )
            )

    # Uniqueness check
    seen: dict[str, int] = {}
    for inv in invariants:
        if inv.id in seen:
            raise ExtractorError(
                f"Duplicate invariant id {inv.id!r}: first seen at byte "
                f"offset {seen[inv.id]}, again at {inv.source_offset}. "
                f"IDs must be unique across the entire spec."
            )
        seen[inv.id] = inv.source_offset

    return invariants


class Stub(TypedDict):
    id: str
    note: str
    pointer: str | None


STUBS: list[Stub] = [
    {
        "id": "github-token-scope-minimum-viable",
        "note": (
            "This invariant is a deployment/configuration rule enforced by "
            "GitHub itself, not by code. Moved to `docs/deployment.md` \u2014 "
            "kept here as a stub for citability."
        ),
        "pointer": "See `docs/deployment.md` for the enforcement story.",
    },
    {
        "id": "llm-output-is-untrusted",
        "note": (
            "Dropped as a standalone invariant. This is the umbrella framing "
            "for `severity-set-by-policy`, `evidence-tier-schema-enforced`, "
            "and `confidence-is-computed-not-assigned`. The child entries "
            "are actionable; the umbrella is not."
        ),
        "pointer": (
            "See the three child entries and `docs/trust-boundaries.md` section 6 for the framing."
        ),
    },
    {
        "id": "prompt-caching-always-on",
        "note": (
            "Dropped as an invariant. Prompt caching is a performance "
            "convention (violation is expensive, not incorrect). The "
            "canonical record lives in `DECISIONS.md#013` point 4 "
            '("Prompt caching. Enabled by default per the existing '
            'prompt-caching-always-on convention.") and `docs/spec.md` '
            "§9.5. V1 ships single-cacheable-block packing: "
            "cross-call-stable content goes in `system_prompt` with "
            "`cache_control: ephemeral`; per-call content goes in "
            "`user_prompt` outside the cache boundary. For analyze "
            "(the `analyze-v4` cache-packing repartition), stable means "
            "CROSS-FILE stable — the invariant prefix "
            "(`SYSTEM_PROMPT_STABLE_PREFIX`) is byte-identical across "
            "every file in a review; per-file scope context travels in "
            "`user_prompt`. Prompts below the model's min-cacheable "
            "floor (`llm/pricing.py::MIN_CACHEABLE_TOKENS`) silently "
            "skip caching. V1.5+ extends to multi-block messages with "
            "per-block `cache_control` on stable file-context user "
            "blocks (deferred until `LLMRequest.messages` becomes "
            "supported). Wrapper-side default: "
            "`LLMRequest.cache_control: bool = True`."
        ),
        "pointer": "See `DECISIONS.md#013` point 4 and `docs/spec.md` §9.5.",
    },
]


HEADER = """# Invariants

<!-- Generated from docs/spec.md by scripts/extract_invariants.py.
     Do not edit directly. To change an invariant, edit the invariant tag
     block in the cited spec section and regenerate this file.

     Labels:
       [security-critical] - violation is a security bug, not just a logic bug
     Fields:
       Rule                 - what must always be true
       Violation looks like - concrete patterns to grep for or flag in review
       Check                - (optional) concrete grep or test command that
                               verifies the invariant from the command line
-->

"""


def render(invariants: list[Invariant]) -> str:
    """Render the sorted invariant catalog plus forwarding stubs."""
    # Merge real invariants with stubs, sort alphabetically by id
    all_entries: list[tuple[str, str]] = []

    for inv in invariants:
        all_entries.append((inv.id, render_entry(inv)))

    for stub in STUBS:
        all_entries.append((stub["id"], render_stub(stub)))

    all_entries.sort(key=lambda x: x[0])

    out = [HEADER]
    for _, rendered in all_entries:
        out.append(rendered)
        out.append("---\n")

    # Strip trailing separator
    if out[-1] == "---\n":
        out.pop()

    return "\n".join(out)


def render_entry(inv: Invariant) -> str:
    label = " `[security-critical]`" if inv.security_critical else ""
    lines = [
        f"## {inv.id}{label}",
        "",
        f"**Source.** \u00a7{inv.section_num} {inv.section_title}",
        "",
        f"**Rule.** {inv.rule}",
        "",
        f"**Violation looks like.** {inv.violation}",
        "",
    ]
    if inv.check:
        lines.append(f"**Check.** {inv.check}")
        lines.append("")
    return "\n".join(lines)


def render_stub(stub: Stub) -> str:
    lines = [
        f"## {stub['id']}",
        "",
        f"*Note: {stub['note']}*",
        "",
    ]
    if stub["pointer"]:
        lines.append(stub["pointer"])
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed invariants.md matches what extraction "
        "would produce. Exit 1 on mismatch; write nothing.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=SPEC_PATH,
        help=f"Path to spec file (default: {SPEC_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help=f"Path to output file (default: {OUT_PATH})",
    )
    args = parser.parse_args()

    try:
        spec_text = args.spec.read_text()
    except FileNotFoundError:
        print(f"error: spec file not found: {args.spec}", file=sys.stderr)
        return 2

    try:
        invariants = extract(spec_text)
    except ExtractorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    generated = render(invariants)

    if args.check:
        if not args.out.exists():
            print(
                f"error: {args.out} does not exist; run without --check to create it",
                file=sys.stderr,
            )
            return 1
        committed = args.out.read_text()
        if committed != generated:
            print(
                f"error: {args.out} is out of sync with {args.spec}. "
                f"Run `python {sys.argv[0]}` to regenerate, then commit.",
                file=sys.stderr,
            )
            return 1
        print(
            f"ok: {args.out} matches extraction from {args.spec} "
            f"({len(invariants)} live invariants, {len(STUBS)} stubs)"
        )
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(generated)
    print(f"wrote {args.out}: {len(invariants)} live invariants, {len(STUBS)} stubs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
