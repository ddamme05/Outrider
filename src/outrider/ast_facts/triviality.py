# Per specs/2026-06-10-trivial-scope-filter.md — the trivial-scope classifier.
# See DECISIONS.md#044 (veto shares the prompt's clipping frame) and
# DECISIONS.md#018 Amended 2026-06-11 (SkipReason.ALL_SCOPES_TRIVIAL).
"""Ordinary-comment-only change classification for the trivial-scope filter.

A changed scope classifies TRIVIAL iff every changed line (head-side
added, base-side kept-removed) is a standalone ordinary-comment line:
non-empty non-whitespace content entirely contained within tree-sitter
comment-node spans, firing none of the directive matcher rules. Every
error path fails closed to NON_TRIVIAL — the filter can only under-skip.

Two-step API so analyze pays one parse per file side, not per scope:
`build_triviality_context(...)` parses head (and base when present) and
precomputes the per-line tables; `classify_scope_triviality(...)` is
then pure per-scope work over `coordinates.ScopeChangedLineSpans`.

Raw tree-sitter stays inside this module per the AST firewall; the
matcher rules are text rules over comment bodies, versioned by
`TRIVIAL_FILTER_VERSION` — any rule or denylist change bumps it (the
version is an audit-event field and a planned lever-#8 cache-key
component, FUP-166).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, Literal

import tree_sitter_python
from pydantic import BaseModel, ConfigDict, Field
from tree_sitter import Language, Parser

from outrider.ast_facts.models import TrivialityReason

if TYPE_CHECKING:
    from collections.abc import Callable

    from tree_sitter import Node

    from outrider.coordinates.spans import ScopeChangedLineSpans

_PY_LANGUAGE: Final = Language(tree_sitter_python.language())
_PARSER: Final = Parser(_PY_LANGUAGE)

# Bump on ANY change to the matcher rules or denylist below — the version
# is recorded on every ScopeExclusionEvent and becomes a lever-#8
# cache-key component (FUP-166). Code-pinned by design: never injectable.
TRIVIAL_FILTER_VERSION: Final = "trivial-filter-v1"

# ---------------------------------------------------------------------------
# Matcher constants (pinned by the spec; structural tests pin each entry)
# ---------------------------------------------------------------------------

# Shape rule: identifier + optional [bracket-group] + ':' or '='. The '.'
# and '-' in the char class are load-bearing (SPDX-License-Identifier:,
# nuitka-project:, pyre-fixme[7]:) — NOT a Python-identifier predicate.
_SHAPE_RE: Final = re.compile(r"^[A-Za-z_][\w.-]*\s*(\[[^\]]*\]\s*)?[:=]")

# Modeline rule: version qualifier is load-bearing — vim honors vim600:,
# vim<800:, vim>702:, vim=701:, and the </> forms escape every other rule.
_MODELINE_RE: Final = re.compile(r"(^|\s)(vi|vim|Vim|ex)[<=>]?\d*:")

# Emacs trailer phrases (vim window is first/last 5 lines; emacs scans
# the last page, ~3000 chars).
_EMACS_PHRASE_RE: Final = re.compile(r"local variables:|^end:", re.IGNORECASE)
_MODELINE_WINDOW_LINES: Final = 5
_EMACS_WINDOW_BYTES: Final = 3000

# Magic prefixes: shebang, emacs file-locals/PEP 263 form, notebook cell
# markers / escaped magics, PEP 723 fence, Sphinx #: doc-comments,
# @generated-class review markers.
_MAGIC_PREFIXES: Final = ("!", "-*-", "%", "///", ":", "@")

# Bare-token denylist, casefolded first-word match (bracket-tolerant).
_BARE_TOKENS_CASEFOLD: Final = frozenset(
    {"noqa", "nosec", "nosemgrep", "noinspection", "nopep8", "noreorder", "sourcery"}
)
_BARE_PREFIXES_CASEFOLD: Final = ("pyre-",)
# Exact-case first-word: Databricks emits these uppercase-only; casefold
# would make prose ("# Command line args...") non-trivial and bias the
# flip-gate trivial-share measurement.
_BARE_TOKENS_EXACT: Final = frozenset({"MAGIC", "COMMAND"})
_EXACT_PHRASES: Final = ("Databricks notebook source",)

# Unanchored inner scan — deliberately tiny. flake8 honors a hash+noqa
# marker anywhere in the physical line (we over-match the bare token as
# the cheaper conservative form); pylint's pragma parser matches its
# "pylint" + colon marker after arbitrary leading text.
_INNER_NOQA_RE: Final = re.compile(r"noqa", re.IGNORECASE)
_INNER_PYLINT_RE: Final = re.compile(r"pylint:")

# PEP 723 inline script metadata: opener / content / closer line shapes.
# Close semantics are GREEDY (last closer in the content run) per the
# PEP's reference regex; an unclosed opener protects to EOF.
_PEP723_OPENER_RE: Final = re.compile(r"^# /// (?P<type>[a-zA-Z0-9-]+)\s*$")
_PEP723_CONTENT_RE: Final = re.compile(r"^#(| .*)$")
_PEP723_CLOSER: Final = "# ///"

_FIRST_WORD_RE: Final = re.compile(r"[^\s\[]+")


class TrivialityVerdict(BaseModel):
    """The classifier's per-scope answer. `trivial=True` only with
    reason `ALL_LINES_ORDINARY_COMMENT`; every other reason is a veto
    naming the first offending side/line for the audit event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trivial: bool
    reason: TrivialityReason
    offending_side: Literal["head", "base"] | None = None
    offending_line: int | None = None
    filter_version: str = TRIVIAL_FILTER_VERSION


class SideTable(BaseModel):
    """Per-side per-line classification tables, computed once per file.

    `comment_lines` maps 1-indexed line numbers to RAW line text for
    lines whose non-whitespace content is non-empty and entirely within
    comment-node spans (the only trivial-eligible lines).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    parse_ok: bool
    total_lines: int = Field(ge=0)
    comment_lines: dict[int, str]
    blank_lines: frozenset[int]
    pep723_lines: frozenset[int]
    emacs_window_start_line: int = Field(ge=1)


class FileTrivialityContext(BaseModel):
    """Both sides' tables. `base=None` iff no base content was supplied
    (added-status files); the classifier fails closed if base-side
    changed lines then appear."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    head: SideTable
    base: SideTable | None


# ---------------------------------------------------------------------------
# Context construction (one parse per side per file)
# ---------------------------------------------------------------------------


def build_triviality_context(
    head_source: bytes, base_source: bytes | None
) -> FileTrivialityContext:
    """Parse each available side once and precompute the line tables."""
    return FileTrivialityContext(
        head=_build_side_table(head_source),
        base=_build_side_table(base_source) if base_source is not None else None,
    )


def _comment_spans(root: Node) -> list[tuple[int, int]]:
    """Byte spans of every comment node, via cursor walk (comments are
    leaves; Python comments never span lines)."""
    spans: list[tuple[int, int]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "comment":
            spans.append((node.start_byte, node.end_byte))
            continue
        stack.extend(node.children)
    return spans


def _build_side_table(source: bytes) -> SideTable:
    tree = _PARSER.parse(source)
    parse_ok = not tree.root_node.has_error

    # Line table: line_starts[k-1] = byte offset where 1-indexed line k begins.
    line_starts = [0]
    for i, byte in enumerate(source):
        if byte == 0x0A:
            line_starts.append(i + 1)
    total_lines = len(line_starts)

    def line_bytes(line_no: int) -> bytes:
        start = line_starts[line_no - 1]
        end = line_starts[line_no] if line_no < total_lines else len(source)
        return source[start:end]

    comment_spans = sorted(_comment_spans(tree.root_node)) if parse_ok else []

    comment_lines: dict[int, str] = {}
    blank_lines: set[int] = set()
    for line_no in range(1, total_lines + 1):
        raw = line_bytes(line_no)
        if not raw.strip():
            blank_lines.add(line_no)
            continue
        if not parse_ok:
            continue
        # Non-whitespace content must sit entirely within comment spans:
        # node-span containment, never a lstrip().startswith('#') heuristic
        # ('#' inside a string literal is the pinned counterexample).
        line_start_byte = line_starts[line_no - 1]
        content_start = line_start_byte + (len(raw) - len(raw.lstrip()))
        content_end = line_start_byte + len(raw.rstrip())
        if any(s <= content_start and content_end <= e for s, e in comment_spans):
            comment_lines[line_no] = raw.decode("utf-8", errors="replace")

    pep723_lines = _pep723_protected_lines(line_bytes, total_lines)
    emacs_window_start = 1
    if len(source) > _EMACS_WINDOW_BYTES:
        threshold = len(source) - _EMACS_WINDOW_BYTES
        emacs_window_start = total_lines
        for line_no in range(1, total_lines + 1):
            if line_starts[line_no - 1] >= threshold:
                emacs_window_start = line_no
                break

    return SideTable(
        parse_ok=parse_ok,
        total_lines=total_lines,
        comment_lines=comment_lines,
        blank_lines=frozenset(blank_lines),
        pep723_lines=frozenset(pep723_lines),
        emacs_window_start_line=emacs_window_start,
    )


def _pep723_protected_lines(line_bytes: Callable[[int], bytes], total_lines: int) -> set[int]:
    """Line numbers inside PEP 723 metadata blocks, fences inclusive.

    Greedy close: within the run of content-shaped lines after an opener,
    the block closes at the LAST '# ///' line (an interior '# ///' is
    valid content and must not close early — a decoy early-closer would
    strand later dependency lines as 'prose'). An unclosed opener
    protects everything to EOF.
    """
    texts = {
        n: line_bytes(n).decode("utf-8", errors="replace").rstrip("\r\n")
        for n in range(1, total_lines + 1)
    }
    protected: set[int] = set()
    line_no = 1
    while line_no <= total_lines:
        if not _PEP723_OPENER_RE.match(texts[line_no]):
            line_no += 1
            continue
        last_closer: int | None = None
        probe = line_no + 1
        while probe <= total_lines and _PEP723_CONTENT_RE.match(texts[probe]):
            if texts[probe].rstrip() == _PEP723_CLOSER:
                last_closer = probe
            probe += 1
        block_end = last_closer if last_closer is not None else total_lines
        protected.update(range(line_no, block_end + 1))
        line_no = max(block_end + 1, probe)
    return protected


# ---------------------------------------------------------------------------
# The directive matcher (rules over one standalone comment line)
# ---------------------------------------------------------------------------


def _normalize_body(raw_line: str) -> str:
    """Body = comment text after stripping ALL leading '#' then leading
    horizontal whitespace ('# fmt: off' / '## type: ignore' / '#:' →
    'fmt: off' / 'type: ignore' / ':')."""
    text = raw_line.strip()
    text = text.lstrip("#")
    return text.lstrip(" \t")


def _is_directive(raw_line: str, line_no: int, side: SideTable) -> bool:
    """True iff any matcher rule fires — the line is NOT ordinary prose.

    Rule order per the spec; every rule fails closed (firing on prose is
    the safe error). Do NOT generalize the inner scan to 'any identifier
    followed by whitespace' — that matches all prose and collapses the
    filter into never-skip.
    """
    stripped = raw_line.strip()
    body = _normalize_body(raw_line)

    # Rule 1 — positional, lines 1-2: PEP 263 cookies are SEARCHED within
    # the comment (cpython tokenize.cookie_re), so any line-1/2 comment
    # fails closed. Defense-in-depth: admission structure (no module
    # scope units) is the primary protection today.
    if line_no <= 2:
        return True

    # Rule 2 — modelines (first/last 5 lines) + emacs trailer (last page).
    in_modeline_window = (
        line_no <= _MODELINE_WINDOW_LINES or line_no > side.total_lines - _MODELINE_WINDOW_LINES
    )
    if in_modeline_window and _MODELINE_RE.search(body):
        return True
    if line_no >= side.emacs_window_start_line and _EMACS_PHRASE_RE.search(body):
        return True

    # Rule 3 — PEP 723 blocks (fences inclusive; greedy close; unclosed→EOF).
    if line_no in side.pep723_lines:
        return True

    # Rule 4 — shape: identifier(+bracket)[:=]; unknown tools fail closed.
    if _SHAPE_RE.match(body):
        return True

    # Rule 5 — magic prefixes; plus the Sphinx '#:' raw form (stripping
    # all '#' from '#: doc' leaves ': doc', already covered by the ':'
    # prefix — the raw-form check is belt-and-suspenders for '#:doc').
    if body.startswith(_MAGIC_PREFIXES) or stripped.startswith("#:"):
        return True

    # Rule 6 — bare-token denylist (three pinned match modes).
    first_word_match = _FIRST_WORD_RE.match(body)
    if first_word_match:
        first_word = first_word_match.group(0)
        folded = first_word.casefold()
        if folded in _BARE_TOKENS_CASEFOLD or folded.startswith(_BARE_PREFIXES_CASEFOLD):
            return True
        if first_word in _BARE_TOKENS_EXACT:
            return True
    if any(body.startswith(phrase) for phrase in _EXACT_PHRASES):
        return True

    # Rule 7 — unanchored inner scan (tiny, explicit set).
    return bool(_INNER_NOQA_RE.search(body) or _INNER_PYLINT_RE.search(body))


# ---------------------------------------------------------------------------
# Per-scope classification
# ---------------------------------------------------------------------------


def classify_scope_triviality(
    changed: ScopeChangedLineSpans, context: FileTrivialityContext
) -> TrivialityVerdict:
    """Classify one admitted scope's changed lines. Fail-closed: any
    uncertainty, parse error, missing base, blank line, non-comment
    content, or directive yields NON_TRIVIAL."""
    if not changed.head_added and not changed.base_removed:
        return _veto(TrivialityReason.NO_CHANGED_LINES)
    if not context.head.parse_ok:
        return _veto(TrivialityReason.PARSE_ERROR)
    if changed.base_removed:
        if context.base is None:
            return _veto(TrivialityReason.MISSING_BASE_CONTENT)
        if not context.base.parse_ok:
            return _veto(TrivialityReason.PARSE_ERROR, side="base")

    sides: tuple[tuple[Literal["head", "base"], SideTable | None, tuple[int, ...]], ...] = (
        ("head", context.head, tuple(e.line_no for e in changed.head_added)),
        ("base", context.base, tuple(e.line_no for e in changed.base_removed)),
    )
    for side_name, table, line_nos in sides:
        if table is None or not line_nos:
            continue
        for line_no in line_nos:
            if line_no in table.blank_lines:
                return _veto(
                    TrivialityReason.BLANK_OR_WHITESPACE_LINE, side=side_name, line=line_no
                )
            raw = table.comment_lines.get(line_no)
            if raw is None:
                return _veto(TrivialityReason.NON_COMMENT_CONTENT, side=side_name, line=line_no)
            if _is_directive(raw, line_no, table):
                return _veto(TrivialityReason.DIRECTIVE_COMMENT, side=side_name, line=line_no)
    return TrivialityVerdict(trivial=True, reason=TrivialityReason.ALL_LINES_ORDINARY_COMMENT)


def _veto(
    reason: TrivialityReason,
    *,
    side: Literal["head", "base"] | None = None,
    line: int | None = None,
) -> TrivialityVerdict:
    return TrivialityVerdict(trivial=False, reason=reason, offending_side=side, offending_line=line)
