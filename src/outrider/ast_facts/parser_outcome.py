# Skip-classification helpers per
# specs/2026-04-30-ast-facts-module.md Internal contracts.
"""Classify parser exclusions before decoding or parsing source.

should_skip first rejects invalid path forms, then evaluates EXCLUSION_RULES
in declared order. The first match wins: size, vendored paths, generated or
minified filenames, then the raw-byte banner. No tree-sitter import is needed.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Final, cast

from outrider.ast_facts.models import ExclusionRule, SkipReason

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MAX_PARSE_BYTES: Final[int] = 1_048_576  # 1 MiB; V1 DoS guard

# Order IS precedence: the first matching rule wins, so size runs before the
# vendored path prefixes, those before the generated and minified filename
# suffixes, and the content banner runs last.
EXCLUSION_RULES: Final[tuple[ExclusionRule, ...]] = (
    ExclusionRule(reason=SkipReason.OVERSIZED, kind="size", pattern=MAX_PARSE_BYTES),
    ExclusionRule(reason=SkipReason.VENDORED, kind="path_prefix", pattern="vendor/"),
    ExclusionRule(reason=SkipReason.VENDORED, kind="path_prefix", pattern="node_modules/"),
    ExclusionRule(reason=SkipReason.VENDORED, kind="path_prefix", pattern="third_party/"),
    ExclusionRule(reason=SkipReason.VENDORED, kind="path_prefix", pattern=".venv/"),
    ExclusionRule(reason=SkipReason.VENDORED, kind="path_prefix", pattern="venv/"),
    ExclusionRule(reason=SkipReason.GENERATED_FILENAME, kind="filename_suffix", pattern="_pb2.py"),
    ExclusionRule(
        reason=SkipReason.GENERATED_FILENAME, kind="filename_suffix", pattern="_pb2_grpc.py"
    ),
    ExclusionRule(reason=SkipReason.GENERATED_FILENAME, kind="filename_suffix", pattern=".pyi"),
    # TypeScript declaration files — the `.pyi` analog, all three emit variants.
    ExclusionRule(reason=SkipReason.GENERATED_FILENAME, kind="filename_suffix", pattern=".d.ts"),
    ExclusionRule(reason=SkipReason.GENERATED_FILENAME, kind="filename_suffix", pattern=".d.mts"),
    ExclusionRule(reason=SkipReason.GENERATED_FILENAME, kind="filename_suffix", pattern=".d.cts"),
    ExclusionRule(reason=SkipReason.MINIFIED, kind="filename_suffix", pattern=".min.py"),
    ExclusionRule(reason=SkipReason.MINIFIED, kind="filename_suffix", pattern=".min.js"),
    ExclusionRule(reason=SkipReason.MINIFIED, kind="filename_suffix", pattern=".min.mjs"),
    ExclusionRule(reason=SkipReason.MINIFIED, kind="filename_suffix", pattern=".min.cjs"),
    ExclusionRule(reason=SkipReason.GENERATED_BANNER, kind="banner", pattern=b"DO NOT EDIT"),
)

# Banner check inspects only the first N bytes of source to keep the
# check O(1) regardless of file size; size-check runs before this so
# oversized files never reach the banner check.
_BANNER_PREFIX_BYTES: Final[int] = 200


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------


def should_skip(file_path: str, source: bytes) -> SkipReason | None:
    """Classify a file against `EXCLUSION_RULES` in declared order.

    Returns the first matched rule's `reason`, or `None` if no rule
    matches. The ordering is the precedence order per Internal
    contracts: size first (cheapest, no source inspection beyond
    `len(source)`), then path-pattern, then filename-suffix, then
    content-pattern (raw-byte check on first 200 bytes — no UTF-8
    decode, so this stays cheap on non-UTF-8 source).
    """
    # Literal prefix rules require POSIX repo-relative paths (trust-boundary #5).
    # coordinates owns validation; reject forms that would bypass these rules.
    if file_path.startswith("/") or "\\" in file_path:
        msg = (
            f"file_path must be POSIX repo-relative per trust-boundary #5; "
            f"got {file_path!r}. `coordinates/` is responsible for normalization "
            f"before paths reach `ast_facts/`."
        )
        raise ValueError(msg)
    # PurePosixPath here is used only to extract the trailing filename
    # component for suffix matching; the prefix check above guards the
    # `path_prefix` rule's literal-startswith semantics.
    relative = PurePosixPath(file_path)
    name = relative.name

    # Pydantic's cross-field validator on ExclusionRule already enforces
    # the kind/pattern type pairing at construction time, so the casts
    # below are typing-only narrowing for mypy/ruff.
    for rule in EXCLUSION_RULES:
        # `>` per canonical §5.5 wording ("source byte length exceeding
        # `MAX_PARSE_BYTES`") and the approved ast_facts/ spec
        # (`len(source) > MAX_PARSE_BYTES`). A file of exactly
        # MAX_PARSE_BYTES bytes (1 MiB) passes through. If the boundary
        # case ever needs OVERSIZED treatment, that's a canonical
        # amendment routed via DECISIONS.md, not a silent code change.
        if rule.kind == "size" and len(source) > cast("int", rule.pattern):
            return rule.reason
        if rule.kind == "path_prefix" and file_path.startswith(cast("str", rule.pattern)):
            return rule.reason
        # Case-insensitive: registry dispatch deliberately lowercases
        # extensions (`Foo.PY`, `JQUERY.MIN.JS` are legal on
        # case-insensitive filesystems), so the suffix skips must match
        # the same set of files the adapters would otherwise parse.
        # Patterns in EXCLUSION_RULES are lowercase by construction.
        if rule.kind == "filename_suffix" and name.lower().endswith(cast("str", rule.pattern)):
            return rule.reason
        # Case-insensitive raw-byte check per Internal contracts.
        # `bytes.upper()` works on ASCII subsequences without UTF-8
        # decode, preserving size-and-pattern-before-decode ordering.
        #
        # UTF-16/UTF-32 banners contain interleaved NUL bytes and do not match.
        # Those files fail the later UTF-8 decode and remain audited as failed.
        if (
            rule.kind == "banner"
            and cast("bytes", rule.pattern) in source[:_BANNER_PREFIX_BYTES].upper()
        ):
            return rule.reason

    return None
