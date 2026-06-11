"""Structural eval scenario: the trivial-scope classifier and its directive matcher.

Per specs/2026-06-10-trivial-scope-filter.md: LLM-free validation of
`ast_facts.triviality` — the comment-only containment rule (node-span
based, both sides), the fail-closed paths, and every pinned matcher rule
with one structural test per denylist token. Tests exercise RAW comment
lines (the full `# …` text) through the public classify API, never
pre-stripped bodies, so a wrong body-extraction step fails here.
"""

from __future__ import annotations

import pytest
from unidiff import PatchSet

from outrider.ast_facts import (
    TRIVIAL_FILTER_VERSION,
    TrivialityReason,
    build_triviality_context,
    classify_scope_triviality,
)
from outrider.ast_facts.models import ScopeUnit
from outrider.coordinates import (
    ChangedLineSpan,
    ScopeChangedLineSpans,
    changed_line_spans,
    line_range_to_span,
)

# ---------------------------------------------------------------------------
# Helpers — build a head file with one interesting comment line inside a
# function body, classify a scope whose only changed line is that line.
# ---------------------------------------------------------------------------


def _head_with_comment_at(
    raw_line: str, *, at_line: int = 4, total_lines: int = 8
) -> tuple[str, int]:
    """A valid module: `def f():` + body of pad comments, with `raw_line`
    (indented into the body) at 1-indexed `at_line`. Returns (source, line)."""
    assert at_line >= 3, "lines 1-2 are def/pass; use _head_starting_with_comments"
    lines = ["def f():", "    pass"]
    while len(lines) < at_line - 1:
        lines.append("    # pad")
    lines.append(f"    {raw_line.strip()}" if raw_line.strip() else raw_line)
    while len(lines) < total_lines:
        lines.append("    # pad")
    return "\n".join(lines) + "\n", at_line


def _classify_added(raw_line: str, *, at_line: int = 4, total_lines: int = 8):
    head, line_no = _head_with_comment_at(raw_line, at_line=at_line, total_lines=total_lines)
    ctx = build_triviality_context(head.encode(), None)
    changed = ScopeChangedLineSpans(
        head_added=(
            ChangedLineSpan(line_no=line_no, span=line_range_to_span(line_no, line_no, head)),
        ),
        base_removed=(),
    )
    return classify_scope_triviality(changed, ctx)


# ---------------------------------------------------------------------------
# The containment rule (node-span based, never a line heuristic)
# ---------------------------------------------------------------------------


def test_ordinary_prose_comment_is_trivial() -> None:
    verdict = _classify_added("# explains the retry loop below")
    assert verdict.trivial
    assert verdict.reason == TrivialityReason.ALL_LINES_ORDINARY_COMMENT
    assert verdict.filter_version == TRIVIAL_FILTER_VERSION == "trivial-filter-v1"


def test_hash_inside_string_literal_is_not_a_comment() -> None:
    """The pinned counterexample: '#' inside a string is string content,
    not a comment node — a lstrip().startswith('#') heuristic would
    misclassify it."""
    head = 'def f():\n    x = "text # not a comment"\n    pass\n'
    ctx = build_triviality_context(head.encode(), None)
    changed = ScopeChangedLineSpans(
        head_added=(ChangedLineSpan(line_no=2, span=line_range_to_span(2, 2, head)),),
        base_removed=(),
    )
    verdict = classify_scope_triviality(changed, ctx)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.NON_COMMENT_CONTENT


def test_mixed_code_and_trailing_comment_line_is_non_trivial() -> None:
    head = "def f():\n    x = 1  # tweaked trailing comment\n    pass\n"
    ctx = build_triviality_context(head.encode(), None)
    changed = ScopeChangedLineSpans(
        head_added=(ChangedLineSpan(line_no=2, span=line_range_to_span(2, 2, head)),),
        base_removed=(),
    )
    verdict = classify_scope_triviality(changed, ctx)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.NON_COMMENT_CONTENT


def test_blank_line_inside_docstring_fails_closed() -> None:
    """Vacuous-containment hole, closed: a blank changed line inside a
    triple-quoted string changes the runtime value and must veto."""
    head = 'def f():\n    """doc\n\n    tail"""\n    pass\n'
    ctx = build_triviality_context(head.encode(), None)
    changed = ScopeChangedLineSpans(
        head_added=(ChangedLineSpan(line_no=3, span=line_range_to_span(3, 3, head)),),
        base_removed=(),
    )
    verdict = classify_scope_triviality(changed, ctx)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.BLANK_OR_WHITESPACE_LINE


def test_parse_error_fails_closed() -> None:
    head = "def f(:\n    # comment\n"
    ctx = build_triviality_context(head.encode(), None)
    changed = ScopeChangedLineSpans(
        head_added=(ChangedLineSpan(line_no=2, span=line_range_to_span(2, 2, head)),),
        base_removed=(),
    )
    verdict = classify_scope_triviality(changed, ctx)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.PARSE_ERROR


def test_missing_base_with_removed_lines_fails_closed() -> None:
    head = "def f():\n    pass\n    # note\n"
    ctx = build_triviality_context(head.encode(), None)
    changed = ScopeChangedLineSpans(
        head_added=(ChangedLineSpan(line_no=3, span=line_range_to_span(3, 3, head)),),
        base_removed=(ChangedLineSpan(line_no=1, span=line_range_to_span(1, 1, head)),),
    )
    verdict = classify_scope_triviality(changed, ctx)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.MISSING_BASE_CONTENT


def test_no_changed_lines_fails_closed() -> None:
    head = "def f():\n    pass\n"
    ctx = build_triviality_context(head.encode(), None)
    changed = ScopeChangedLineSpans(head_added=(), base_removed=())
    verdict = classify_scope_triviality(changed, ctx)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.NO_CHANGED_LINES


# ---------------------------------------------------------------------------
# Matcher rule 4 — shape (identifier + optional bracket + [:=]); the '.'
# and '-' in the char class are load-bearing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "# type: ignore",
        "# pragma: no cover",
        "# fmt: off",
        "# pylint: disable=missing-docstring",
        "# coding=utf-8",
        "# cython: language_level=3",
        "# SPDX-License-Identifier: MIT",
        "# nuitka-project: --mode=onefile",
        "# pyre-fixme[7]: Expected int",
        "# NOTE: prose with a colon prefix is the accepted v1 cost",
        "# TODO: same accepted cost",
    ],
)
def test_shape_rule_directives_are_non_trivial(raw: str) -> None:
    verdict = _classify_added(raw)
    assert not verdict.trivial, raw
    assert verdict.reason == TrivialityReason.DIRECTIVE_COMMENT


# ---------------------------------------------------------------------------
# Body normalization — strip ALL leading '#', then whitespace. Raw-line
# tests so a wrong extraction step fails here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "## isort: skip_file",  # doubled hash; isort detection is substring-based
        "#fmt: off",  # no space after '#'
        "##noqa",
        "#: Sphinx attribute doc-comment",
    ],
)
def test_body_normalization_catches_hash_variants(raw: str) -> None:
    verdict = _classify_added(raw)
    assert not verdict.trivial, raw
    assert verdict.reason == TrivialityReason.DIRECTIVE_COMMENT


# ---------------------------------------------------------------------------
# Matcher rule 5 — magic prefixes (mid-file; lines 1-2 are rule 1's job)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "# -*- coding: utf-8 -*-",
        "# %%",
        "# %% [markdown]",
        "# %matplotlib inline",
        "# @generated",
        "# /// script",
        "# ///",
    ],
)
def test_magic_prefixes_are_non_trivial(raw: str) -> None:
    verdict = _classify_added(raw)
    assert not verdict.trivial, raw
    assert verdict.reason == TrivialityReason.DIRECTIVE_COMMENT


# ---------------------------------------------------------------------------
# Matcher rule 6 — bare-token denylist: one structural test per token,
# three pinned match modes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "# noqa",
        "# NOQA",  # casefold mode
        "# NoSec",
        "# nosemgrep",
        "# noinspection PyUnresolvedReferences",
        "# nopep8",
        "# noreorder",
        "# sourcery skip: hoist-statement-from-if",
        "# pyre-strict",
        "# pyre-unsafe",
        "# pyre-ignore-all-errors",
        "# pyre-fixme[16]",
        "# pyre-ignore[58]",
        "# MAGIC %sql select 1",  # exact-case mode
        "# COMMAND ----------",
        "# Databricks notebook source",  # exact-phrase mode
    ],
)
def test_bare_token_denylist_is_non_trivial(raw: str) -> None:
    verdict = _classify_added(raw)
    assert not verdict.trivial, raw
    assert verdict.reason == TrivialityReason.DIRECTIVE_COMMENT


@pytest.mark.parametrize(
    "raw",
    [
        "# Command line arguments parsed here",
        "# command to run is documented in the README",
        "# Magic methods overview",
        "# magic numbers explained below",
    ],
)
def test_exact_case_databricks_tokens_do_not_fire_on_prose(raw: str) -> None:
    """The precision pins: casefolding MAGIC/COMMAND would make these
    prose comments non-trivial and bias the flip-gate measurement."""
    verdict = _classify_added(raw)
    assert verdict.trivial, raw


def test_sourcery_prose_collision_is_the_accepted_cost() -> None:
    verdict = _classify_added("# Sourcery is a refactoring tool")
    assert not verdict.trivial  # casefolded first-word; accepted + documented


# ---------------------------------------------------------------------------
# Matcher rule 7 — unanchored inner scan (tiny explicit set)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "# see RFC 1234  # noqa: E501",
        "# removed the NOQA marker from this block",  # over-match, safe direction
        "# pragma pylint: disable=no-member",
    ],
)
def test_inner_scan_catches_embedded_directives(raw: str) -> None:
    verdict = _classify_added(raw)
    assert not verdict.trivial, raw


# ---------------------------------------------------------------------------
# Matcher rules 1-2 — positional: lines 1-2, modeline windows, emacs page
# ---------------------------------------------------------------------------


def test_line_two_comment_fails_closed() -> None:
    """Rule 1: CPython's encoding cookie is SEARCHED within line-1/2
    comments; any comment there fails closed (defense-in-depth — a
    function must start at line 1 for this to even be reachable)."""
    head = "def f():\n    # default text encoding: cp1252\n    pass\n"
    ctx = build_triviality_context(head.encode(), None)
    changed = ScopeChangedLineSpans(
        head_added=(ChangedLineSpan(line_no=2, span=line_range_to_span(2, 2, head)),),
        base_removed=(),
    )
    verdict = classify_scope_triviality(changed, ctx)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.DIRECTIVE_COMMENT


@pytest.mark.parametrize(
    "raw",
    [
        "# vim: set ts=4:",
        "# vim600: set foldmethod=marker:",
        "# vim<800: set noexpandtab:",
        "# vim>702: set foldmethod=expr:",
        "# vim=701: set ts=2:",
        "# edited with vim600: set ts=4:",  # mid-comment form
    ],
)
def test_modelines_in_window_are_non_trivial(raw: str) -> None:
    # Place the line within the LAST 5 lines of an 8-line file.
    verdict = _classify_added(raw, at_line=6, total_lines=8)
    assert not verdict.trivial, raw
    assert verdict.reason == TrivialityReason.DIRECTIVE_COMMENT


def test_version_form_modeline_outside_window_is_ordinary() -> None:
    """vim only honors the first/last `modelines` lines; a version-form
    modeline mid-file is inert for vim, so it may classify ordinary
    (the documented asymmetry — plain `vim:` still hits the shape rule
    anywhere)."""
    verdict = _classify_added("# vim<800: set noexpandtab:", at_line=10, total_lines=30)
    assert verdict.trivial


def test_emacs_local_variables_header_in_last_page_is_non_trivial() -> None:
    """The header escapes the shape rule (two-word phrase); the last-page
    (~3000 chars) phrase scan must catch it even >5 lines from EOF."""
    verdict = _classify_added("# Local Variables:", at_line=10, total_lines=30)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.DIRECTIVE_COMMENT


def test_emacs_phrase_outside_last_page_is_ordinary() -> None:
    """A 'Local Variables:' phrase more than ~3000 bytes from EOF is
    inert for emacs and may classify ordinary."""
    pad = "    # " + "x" * 60
    lines = ["def f():", "    pass", "    # Local Variables:"]
    lines += [pad] * 80  # ~5300 bytes of tail — line 3 is far outside the window
    head = "\n".join(lines) + "\n"
    ctx = build_triviality_context(head.encode(), None)
    changed = ScopeChangedLineSpans(
        head_added=(ChangedLineSpan(line_no=3, span=line_range_to_span(3, 3, head)),),
        base_removed=(),
    )
    verdict = classify_scope_triviality(changed, ctx)
    assert verdict.trivial


# ---------------------------------------------------------------------------
# Matcher rule 3 — PEP 723 blocks: greedy last-closer + unclosed-to-EOF
# ---------------------------------------------------------------------------


def test_pep723_decoy_interior_closer_does_not_strand_dependency_lines() -> None:
    """Greedy close: an interior '# ///' is valid content; the block
    closes at the LAST one, so the evil dependency line after the decoy
    stays protected."""
    lines = [
        "def f():",
        "    pass",
        "# /// script",
        "# dependencies = [",
        "# ///",  # decoy early closer
        '#     "evil-pkg==2.0",',
        "# ]",
        "# ///",  # real closer
        "# unrelated ordinary comment",
    ]
    head = "\n".join(lines) + "\n"
    ctx = build_triviality_context(head.encode(), None)

    def verdict_for(line_no: int):
        changed = ScopeChangedLineSpans(
            head_added=(
                ChangedLineSpan(line_no=line_no, span=line_range_to_span(line_no, line_no, head)),
            ),
            base_removed=(),
        )
        return classify_scope_triviality(changed, ctx)

    evil = verdict_for(6)
    assert not evil.trivial
    assert evil.reason == TrivialityReason.DIRECTIVE_COMMENT
    # The line AFTER the real closer is outside the block — but it's in
    # the last-5 modeline window of this 9-line file, so use the reason
    # only as a not-pep723 proof: it must still be examined as ordinary
    # text by later rules, not auto-protected.
    after = verdict_for(9)
    assert after.reason in (
        TrivialityReason.ALL_LINES_ORDINARY_COMMENT,
        TrivialityReason.DIRECTIVE_COMMENT,
    )


def test_pep723_unclosed_fence_protects_to_eof() -> None:
    lines = [
        "def f():",
        "    pass",
        "# /// script",
        "# dependencies = [",
        '#     "evil-pkg==2.0",',
        "# no closer anywhere",
    ]
    head = "\n".join(lines) + "\n"
    ctx = build_triviality_context(head.encode(), None)
    changed = ScopeChangedLineSpans(
        head_added=(ChangedLineSpan(line_no=5, span=line_range_to_span(5, 5, head)),),
        base_removed=(),
    )
    verdict = classify_scope_triviality(changed, ctx)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.DIRECTIVE_COMMENT


# ---------------------------------------------------------------------------
# Integration: the ride-along deletion veto through the real pipeline
# (patch → coordinates.changed_line_spans → classify) — the spec's
# flagship over-skip seam, end to end.
# ---------------------------------------------------------------------------


def test_ride_along_auth_deletion_vetoes_through_real_pipeline() -> None:
    base = "@require_auth\ndef foo():\n    return 1\n"
    head = "def foo():\n    return 1\n    # auth handled upstream\n"
    patch_text = (
        "--- a/x.py\n+++ b/x.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-@require_auth\n"
        " def foo():\n"
        "     return 1\n"
        "+    # auth handled upstream\n"
    )
    pf = PatchSet.from_string(patch_text)[0]
    su = ScopeUnit(
        unit_id="a" * 64,
        kind="function",
        name="foo",
        qualified_name="x.foo",
        file_path="x.py",
        line_start=1,
        line_end=3,
        byte_start=0,
        byte_end=len(head.encode()),
    )
    changed = changed_line_spans(su, pf, head_source=head, base_source=base)
    ctx = build_triviality_context(head.encode(), base.encode())
    verdict = classify_scope_triviality(changed, ctx)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.NON_COMMENT_CONTENT
    assert verdict.offending_side == "base"
    assert verdict.offending_line == 1


def test_base_side_directive_deletion_vetoes() -> None:
    """Deleting a directive comment (e.g. a noqa) is review-relevant:
    the base side runs the same matcher."""
    base = "def foo():\n    pass\n    # noqa\n"
    head = "def foo():\n    pass\n    # plain note\n"
    patch_text = (
        "--- a/x.py\n+++ b/x.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def foo():\n"
        "     pass\n"
        "-    # noqa\n"
        "+    # plain note\n"
    )
    pf = PatchSet.from_string(patch_text)[0]
    su = ScopeUnit(
        unit_id="b" * 64,
        kind="function",
        name="foo",
        qualified_name="x.foo",
        file_path="x.py",
        line_start=1,
        line_end=3,
        byte_start=0,
        byte_end=len(head.encode()),
    )
    changed = changed_line_spans(su, pf, head_source=head, base_source=base)
    ctx = build_triviality_context(head.encode(), base.encode())
    verdict = classify_scope_triviality(changed, ctx)
    assert not verdict.trivial
    assert verdict.reason == TrivialityReason.DIRECTIVE_COMMENT
    assert verdict.offending_side == "base"


def test_comment_only_modification_is_trivial_through_real_pipeline() -> None:
    """The savings case: replacing one ordinary comment with another —
    both sides comment-only — classifies trivial."""
    # The function starts at line 3 so the changed comment (line 4) is
    # clear of rule 1's lines-1-2 fail-closed window.
    base = "import os\n\ndef foo():\n    # old wording\n    return os.sep\n"
    head = "import os\n\ndef foo():\n    # new wording\n    return os.sep\n"
    patch_text = (
        "--- a/x.py\n+++ b/x.py\n"
        "@@ -1,5 +1,5 @@\n"
        " import os\n"
        " \n"
        " def foo():\n"
        "-    # old wording\n"
        "+    # new wording\n"
        "     return os.sep\n"
    )
    pf = PatchSet.from_string(patch_text)[0]
    su = ScopeUnit(
        unit_id="c" * 64,
        kind="function",
        name="foo",
        qualified_name="x.foo",
        file_path="x.py",
        line_start=3,
        line_end=5,
        byte_start=0,
        byte_end=len(head.encode()),
    )
    changed = changed_line_spans(su, pf, head_source=head, base_source=base)
    ctx = build_triviality_context(head.encode(), base.encode())
    verdict = classify_scope_triviality(changed, ctx)
    assert verdict.trivial
    assert verdict.reason == TrivialityReason.ALL_LINES_ORDINARY_COMMENT
