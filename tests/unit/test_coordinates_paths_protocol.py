"""Unit tests for `coordinates.resolve_candidate_paths` — ImportPathResolver
Protocol implementation per `src/outrider/ast_facts/base.py`.

Per docs/spec.md §10.1 / docs/trust-boundaries.md §5.3 (root-aware surface)
and the ast_facts spec's Protocol contract: relative-only, no `..` traversal,
no shell metacharacters, prefix-validated against `import_root`, and no
symlink components anywhere up to `import_root` (final or ancestor).

See DECISIONS.md#006-two-month-0-spikes-not-five for the test discipline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outrider.coordinates import resolve_candidate_paths

# ----------------------------------------------------------------------------
# Happy path — valid import strings produce both candidates as relative Paths
# ----------------------------------------------------------------------------


def test_simple_import_returns_module_and_package_candidates(tmp_path: Path) -> None:
    """`foo.bar` returns `foo/bar.py` AND `foo/bar/__init__.py` as candidates."""
    result = resolve_candidate_paths("foo.bar", tmp_path)
    assert result == [Path("foo/bar.py"), Path("foo/bar/__init__.py")]


def test_single_part_import_returns_top_level_candidates(tmp_path: Path) -> None:
    """`foo` returns `foo.py` AND `foo/__init__.py`."""
    result = resolve_candidate_paths("foo", tmp_path)
    assert result == [Path("foo.py"), Path("foo/__init__.py")]


def test_deeply_nested_import_returns_nested_candidates(tmp_path: Path) -> None:
    """`a.b.c.d` returns `a/b/c/d.py` AND `a/b/c/d/__init__.py`."""
    result = resolve_candidate_paths("a.b.c.d", tmp_path)
    assert result == [Path("a/b/c/d.py"), Path("a/b/c/d/__init__.py")]


def test_returned_paths_are_relative(tmp_path: Path) -> None:
    """Per the Protocol contract, returned Path objects are repo-relative."""
    result = resolve_candidate_paths("foo.bar", tmp_path)
    # Vacuous-pass guard: if `foo.bar` were ever rejected (returns empty
    # list), the for-loop check would silently pass without testing the
    # relativity contract. Lock that the input produces a meaningful
    # result before checking relativity.
    assert result, (
        "expected non-empty result for 'foo.bar'; otherwise the relativity check below is vacuous"
    )
    for candidate in result:
        assert not candidate.is_absolute()


# ----------------------------------------------------------------------------
# Rejections — empty / malformed import strings
# ----------------------------------------------------------------------------


def test_empty_import_returns_empty_list(tmp_path: Path) -> None:
    """`""` → empty list."""
    assert resolve_candidate_paths("", tmp_path) == []


def test_leading_dot_returns_empty_list(tmp_path: Path) -> None:
    """`".foo"` (split has empty first part) → empty list. Catches relative-import
    leakage that should have been filtered upstream by ast_facts."""
    assert resolve_candidate_paths(".foo", tmp_path) == []


def test_trailing_dot_returns_empty_list(tmp_path: Path) -> None:
    """`"foo."` → empty list."""
    assert resolve_candidate_paths("foo.", tmp_path) == []


def test_consecutive_dots_returns_empty_list(tmp_path: Path) -> None:
    """`"foo..bar"` (empty interior part) → empty list."""
    assert resolve_candidate_paths("foo..bar", tmp_path) == []


def test_triple_dots_return_empty_list(tmp_path: Path) -> None:
    """`"foo...bar"` → empty list (multiple consecutive dots produce empty
    interior parts; rejected by the not-all-parts check, since `.split(".")`
    can never produce a literal `".."` part)."""
    assert resolve_candidate_paths("foo...bar", tmp_path) == []


# ----------------------------------------------------------------------------
# Rejections — forbidden characters in the import string
# ----------------------------------------------------------------------------


def test_forward_slash_in_import_returns_empty(tmp_path: Path) -> None:
    """`"foo/bar"` (path-separator-as-import) → empty list. Python imports
    use `.` as the separator; `/` is malformed."""
    assert resolve_candidate_paths("foo/bar", tmp_path) == []


def test_backslash_in_import_returns_empty(tmp_path: Path) -> None:
    """`"foo\\bar"` → empty list (Windows separator forbidden)."""
    assert resolve_candidate_paths("foo\\bar", tmp_path) == []


@pytest.mark.parametrize(
    "metachar",
    [";", "&", "|", "`", "$", "(", ")", "<", ">", "*", "?", "~", "[", "]", "{", "}", "'", '"'],
)
def test_shell_metacharacter_in_import_returns_empty(metachar: str, tmp_path: Path) -> None:
    """Every shell metacharacter in the import string → empty list."""
    assert resolve_candidate_paths(f"foo{metachar}bar", tmp_path) == []


def test_newline_in_import_returns_empty(tmp_path: Path) -> None:
    """Newline in import string → empty list (header-injection prevention)."""
    assert resolve_candidate_paths("foo\nbar", tmp_path) == []


def test_nul_byte_in_import_returns_empty(tmp_path: Path) -> None:
    """NUL byte in import string → empty list."""
    assert resolve_candidate_paths("foo\x00bar", tmp_path) == []


# ----------------------------------------------------------------------------
# Identifier validation per part — defensive narrowing of LLM-influenced surface
# ----------------------------------------------------------------------------


def test_numeric_prefix_part_returns_empty(tmp_path: Path) -> None:
    """`"foo.123abc"` → empty list. `123abc` is not a valid Python identifier."""
    assert resolve_candidate_paths("foo.123abc", tmp_path) == []


def test_pure_numeric_part_returns_empty(tmp_path: Path) -> None:
    """`"foo.42"` → empty list."""
    assert resolve_candidate_paths("foo.42", tmp_path) == []


def test_keyword_part_returns_empty(tmp_path: Path) -> None:
    """`"foo.class"` → empty list. `class` is a Python keyword, not a module name."""
    assert resolve_candidate_paths("foo.class", tmp_path) == []


def test_dunder_parts_accepted(tmp_path: Path) -> None:
    """`"foo.__init__"` → both candidates returned. Dunders are valid identifiers
    and are common in Python imports (e.g., explicit package init imports)."""
    result = resolve_candidate_paths("foo.__init__", tmp_path)
    assert Path("foo/__init__.py") in result


def test_pycache_part_accepted_but_resolves_normally(tmp_path: Path) -> None:
    """`"foo.__pycache__"` → both candidates returned. The string is a valid
    identifier sequence (`__pycache__` is just an identifier); ast_facts'
    existence check is the gate that prevents `__pycache__` directories from
    being treated as Python modules in practice. The resolver narrows the
    string-validity bar but doesn't impose a Python-import-policy bar."""
    result = resolve_candidate_paths("foo.__pycache__", tmp_path)
    assert Path("foo/__pycache__.py") in result


def test_hyphen_in_part_returns_empty(tmp_path: Path) -> None:
    """`"foo.bar-baz"` → empty list. Hyphens aren't allowed in identifiers."""
    assert resolve_candidate_paths("foo.bar-baz", tmp_path) == []


# ----------------------------------------------------------------------------
# Symlink-component rejection — final component
# ----------------------------------------------------------------------------


def test_final_component_symlink_omits_candidate(tmp_path: Path) -> None:
    """If `foo/bar.py` is a symlink, that candidate is omitted; the
    `__init__.py` candidate (different final component) may still pass.
    """
    foo_dir = tmp_path / "foo"
    foo_dir.mkdir()
    real_file = tmp_path / "real.py"
    real_file.write_text("# real content")
    symlink_target = foo_dir / "bar.py"
    symlink_target.symlink_to(real_file)

    result = resolve_candidate_paths("foo.bar", tmp_path)

    # foo/bar.py is a symlink → omitted
    assert Path("foo/bar.py") not in result
    # foo/bar/__init__.py would require foo/bar/ as a dir; it doesn't exist
    # but isn't a symlink, so it passes the safety check (existence is
    # ast_facts' job).
    assert Path("foo/bar/__init__.py") in result


def test_symlink_pointing_outside_import_root_omitted(tmp_path: Path) -> None:
    """A symlink pointing outside import_root → omitted via the prefix
    validation step (resolved path escapes root).
    """
    # Create a real file outside the import_root.
    outside_root = tmp_path.parent / "outside_repo"
    outside_root.mkdir(exist_ok=True)
    target = outside_root / "secret.py"
    target.write_text("# outside")
    # Inside import_root, place a symlink to that outside file.
    inside = tmp_path / "foo"
    inside.mkdir()
    symlink = inside / "bar.py"
    symlink.symlink_to(target)

    result = resolve_candidate_paths("foo.bar", tmp_path)

    assert Path("foo/bar.py") not in result


# ----------------------------------------------------------------------------
# Symlink-component rejection — ancestor component
# ----------------------------------------------------------------------------


def test_symlinked_import_root_returns_empty(tmp_path: Path) -> None:
    """If `import_root` itself is a symlink, return [] regardless of the
    import string.

    The Protocol contract reads "no path component (final or any ancestor
    up to `import_root`) is a symlink" as INCLUSIVE of root. Without this
    guard, the per-candidate ancestor walk stops one level before root,
    and `python_adapter.resolve_simple_direct_import`'s
    `is_file(follow_symlinks=False)` only protects the FINAL component —
    so a symlinked root would silently be followed at stat time.
    """
    real_root = tmp_path / "real_root"
    real_root.mkdir()
    (real_root / "foo.py").write_text("# real")

    symlinked_root = tmp_path / "symlinked_root"
    symlinked_root.symlink_to(real_root)

    assert resolve_candidate_paths("foo", symlinked_root) == []
    assert resolve_candidate_paths("foo.bar", symlinked_root) == []


def test_ancestor_directory_symlink_omits_candidate(tmp_path: Path) -> None:
    """If `foo/` (a parent component) is a symlink, the `foo/bar.py`
    candidate is omitted — parent-directory-symlink-dodge prevention per
    ast_facts spec line 127.
    """
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    (real_dir / "bar.py").write_text("# inside real_dir")
    # Make foo/ a symlink to real_dir/
    foo_symlink = tmp_path / "foo"
    foo_symlink.symlink_to(real_dir)

    result = resolve_candidate_paths("foo.bar", tmp_path)

    # foo/bar.py would be reachable via the symlink — must be omitted
    # because `foo/` is a symlink even though `foo/bar.py` (its target) is not.
    assert Path("foo/bar.py") not in result
    assert Path("foo/bar/__init__.py") not in result


# ----------------------------------------------------------------------------
# All-clean filesystem — both candidates returned
# ----------------------------------------------------------------------------


def test_no_symlinks_returns_both_candidates(tmp_path: Path) -> None:
    """Plain directory tree with no symlinks → both candidates returned
    (existence-on-disk is not the resolver's concern; ast_facts checks).
    """
    foo = tmp_path / "foo"
    foo.mkdir()
    (foo / "bar.py").write_text("# real")

    result = resolve_candidate_paths("foo.bar", tmp_path)

    assert Path("foo/bar.py") in result
    assert Path("foo/bar/__init__.py") in result


def test_nonexistent_paths_pass_safety_check(tmp_path: Path) -> None:
    """If the candidate path does not exist on disk, the safety check still
    passes (non-existent components are not symlinks). Existence-on-disk
    is ast_facts' job, not the resolver's.
    """
    # tmp_path is empty; no foo/, no bar.py
    result = resolve_candidate_paths("foo.bar", tmp_path)
    assert Path("foo/bar.py") in result
    assert Path("foo/bar/__init__.py") in result


# ----------------------------------------------------------------------------
# Conformance: shape matches the ast_facts Protocol-mock harness
# ----------------------------------------------------------------------------


def test_call_signature_matches_protocol(tmp_path: Path) -> None:
    """The function accepts (import_string: str, import_root: Path) — same
    positional shape ast_facts' Protocol mock asserts via
    `assert_called_once_with("foo.bar", tmp_path)` at
    `tests/unit/test_ast_facts_python.py:339-447`.
    """
    # Calling positionally with the exact shape the Protocol mock uses.
    result = resolve_candidate_paths("foo.bar", tmp_path)
    assert isinstance(result, list)
    # Vacuous-pass guard: `all(isinstance(...) for ...)` returns True
    # for an empty iterable. Without this guard, an `foo.bar`-rejected
    # future would silently pass the element-type assertion below.
    assert result, (
        "expected non-empty result for 'foo.bar'; otherwise the "
        "all-Path check below is vacuous (all() on [] returns True)"
    )
    assert all(isinstance(p, Path) for p in result)


def test_returned_list_can_be_consumed_by_iteration(tmp_path: Path) -> None:
    """Returned list is a real list, iterable in deterministic order, with
    Path elements — matches ast_facts' downstream `for candidate in
    candidates: ...` consumption pattern.
    """
    result = resolve_candidate_paths("foo.bar", tmp_path)
    # Vacuous-pass guard: same risk as the sibling test above. If result
    # were empty, `seen == result` would compare [] to [] and pass
    # without actually exercising iteration.
    assert result, (
        "expected non-empty result for 'foo.bar'; otherwise iteration "
        "is vacuous and seen==result trivially holds"
    )
    seen: list[Path] = []
    for candidate in result:
        seen.append(candidate)
    assert seen == result


# ----------------------------------------------------------------------------
# Idempotence
# ----------------------------------------------------------------------------


def test_repeated_calls_are_idempotent(tmp_path: Path) -> None:
    """Same inputs → equal output across repeated calls. No module-level
    state, no caching with mutable backing."""
    a = resolve_candidate_paths("foo.bar", tmp_path)
    b = resolve_candidate_paths("foo.bar", tmp_path)
    assert a == b


# ----------------------------------------------------------------------------
# import_root edge cases
# ----------------------------------------------------------------------------


def test_import_root_with_trailing_separator_works(tmp_path: Path) -> None:
    """`import_root` passed with or without trailing separator yields the
    same result (Path normalizes)."""
    result_a = resolve_candidate_paths("foo", tmp_path)
    result_b = resolve_candidate_paths("foo", Path(str(tmp_path) + "/"))
    assert result_a == result_b


# ----------------------------------------------------------------------------
# is_valid_import_string predicate — DECISIONS.md#024 point 1 +
# specs/2026-05-23-trace-node.md M3 (NFC normalization)
# ----------------------------------------------------------------------------
#
# These tests pin the predicate behavior INDEPENDENTLY of
# `resolve_candidate_paths` because the predicate is also the shared source
# of truth for `TraceCandidate.import_string` schema-validator (which raises
# rather than returning []). Asymmetric semantics (raise vs return []) make
# round-trip-only tests insufficient — pin the predicate's own contract.


from outrider.coordinates import is_valid_import_string  # noqa: E402


class TestIsValidImportString:
    """Per `DECISIONS.md#024` point 1: predicate raises ValueError on invalid
    input, returns NFC-normalized value on valid. Caller-side semantics
    differ — `resolve_candidate_paths` catches+returns []; schema validators
    let it propagate. The predicate itself raises in both worlds."""

    # Happy path — admits valid forms; returns NFC-normalized value
    def test_simple_dotted_form_returns_unchanged(self) -> None:
        assert is_valid_import_string("foo.bar") == "foo.bar"

    def test_single_part_returns_unchanged(self) -> None:
        assert is_valid_import_string("foo") == "foo"

    def test_deeply_nested_returns_unchanged(self) -> None:
        assert is_valid_import_string("a.b.c.d.e") == "a.b.c.d.e"

    def test_dunder_parts_admit(self) -> None:
        """Dunders like `__init__` ARE valid Python identifiers and admit."""
        assert is_valid_import_string("foo.__init__") == "foo.__init__"

    def test_underscore_prefix_admits(self) -> None:
        assert is_valid_import_string("_private.module") == "_private.module"

    def test_digit_in_middle_admits(self) -> None:
        """Identifier may contain digits after the first character."""
        assert is_valid_import_string("foo2.bar3") == "foo2.bar3"

    # NFC normalization — M3 / adversarial-modeler #1
    def test_nfc_composition_normalizes_decomposed_unicode(self) -> None:
        """Decomposed `é` (e + combining acute U+0065 U+0301) normalizes to
        precomposed `é` (U+00E9). Returned value uses the composed form."""
        decomposed = "café.bar"  # café.bar in NFD
        precomposed = "café.bar"  # café.bar in NFC
        result = is_valid_import_string(decomposed)
        assert result == precomposed

    def test_already_nfc_value_returned_unchanged(self) -> None:
        """NFC normalization is idempotent — pre-normalized input passes through."""
        precomposed = "café.bar"
        assert is_valid_import_string(precomposed) == precomposed

    def test_homoglyph_passes_through_consistently(self) -> None:
        """Per M3: NFC is composition normalization, NOT transliteration.
        Cyrillic `а` U+0430 is a valid Python identifier and stays Cyrillic.
        The predicate's job is CONSISTENCY (same form in/out), not rejection."""
        cyrillic_a = "а"  # Cyrillic small letter a
        # Confirm Python identifier check admits Cyrillic (precondition; if
        # this changes, the homoglyph-consistency story changes too)
        assert cyrillic_a.isidentifier()
        result = is_valid_import_string(f"foo.{cyrillic_a}bc")
        assert result == f"foo.{cyrillic_a}bc"

    def test_homoglyph_produces_distinct_candidate_id(self) -> None:
        """Sharp-edges F4: Latin `a` vs Cyrillic `а` produce DISTINCT
        `compute_candidate_id` outputs. Pins that future
        transliteration/case-folding (if any caller adds one) would
        loudly fail this test rather than silently collapse the two
        IDs."""
        from outrider.policy.canonical import compute_candidate_id

        latin = is_valid_import_string("foo.abc")  # Latin a-b-c
        cyrillic = is_valid_import_string("foo.аbc")  # Cyrillic а, Latin b-c
        # Sanity precondition: the two strings differ at the byte level
        assert latin != cyrillic
        # Sanity precondition: both produced canonical (NFC) forms
        assert latin == "foo.abc"
        # candidate_id derivations must differ — otherwise homoglyph
        # candidates would collapse on the dedup-by-candidate_id reducer
        id_latin = compute_candidate_id(
            source_proposal_hash="a" * 64, import_string=latin, reason="r"
        )
        id_cyrillic = compute_candidate_id(
            source_proposal_hash="a" * 64, import_string=cyrillic, reason="r"
        )
        assert id_latin != id_cyrillic

    # Trojan-Source defense — sharp-edges F3 + CVE-2021-42574
    # `validate_diff_path` rejects bidi-override / invisible-format chars.
    # `is_valid_import_string` honors the same defense per the audit-shadow
    # promise — without it, U+200D (ZWJ, `Other_ID_Continue` in Unicode 16)
    # would pass `str.isidentifier()` and embed in an import string that
    # displays differently from the bytes operators see.

    @pytest.mark.parametrize(
        "trojan_codepoint",
        [
            "​",  # Zero Width Space
            "‎",  # LTR Mark
            "‏",  # RTL Mark
            "‪",  # LRE
            "‫",  # RLE
            "‬",  # PDF
            "‭",  # LRO
            "‮",  # RLO (CVE-2021-42574 canonical)
            "⁦",  # LRI
            "⁧",  # RLI
            "⁨",  # FSI
            "⁩",  # PDI
            "﻿",  # BOM / ZWNBSP
        ],
    )
    def test_trojan_source_codepoints_rejected(self, trojan_codepoint: str) -> None:
        """Every codepoint in `_TROJAN_SOURCE_CHARS_RE` is rejected by
        the predicate, matching `validate_diff_path`'s audit-shadow rule.
        Construct `foo<codepoint>.bar` so the codepoint embeds inside an
        identifier (otherwise some codepoints would fail `isidentifier()`
        independently and the rejection path would be ambiguous)."""
        value = f"foo{trojan_codepoint}bar.baz"
        with pytest.raises(ValueError, match="bidi-override or invisible-format characters"):
            is_valid_import_string(value)

    def test_zwj_and_zwnj_deliberately_admitted(self) -> None:
        """U+200C (ZWNJ) and U+200D (ZWJ) are DELIBERATELY excluded from
        `_TROJAN_SOURCE_CHARS_RE` (see `diff_parser.py:51-56` rationale:
        legitimate use in Persian/Arabic word-joining + Hindi/Devanagari
        conjuncts + emoji ZWJ sequences). They ARE `Other_ID_Continue` in
        Unicode 16 so `str.isidentifier()` admits them, AND the predicate
        admits them by design — `validate_diff_path` makes the same
        trade-off. Rejecting them would block legitimate non-Latin-script
        identifier contributions, which the project deliberately accepts
        instead of the marginal homoglyph-attack risk."""
        # ZWJ-embedded identifier passes (precondition)
        assert "foo‍bar".isidentifier()
        # Predicate ADMITS (this is the documented trade-off)
        result = is_valid_import_string("foo‍bar.baz")
        assert result == "foo‍bar.baz"

    # Rejections — each raises ValueError; message discriminates the reason
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            is_valid_import_string("")

    def test_backslash_raises(self) -> None:
        with pytest.raises(ValueError, match="path separators"):
            is_valid_import_string("foo\\bar")

    def test_forward_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="path separators"):
            is_valid_import_string("foo/bar")

    @pytest.mark.parametrize(
        "value",
        [
            "foo;bar",
            "foo&bar",
            "foo|bar",
            "foo`bar",
            "foo$bar",
            "foo(bar",
            "foo>bar",
            "foo*bar",
            "foo?bar",
            "foo\nbar",
            "foo\x00bar",
        ],
    )
    def test_shell_metacharacter_raises(self, value: str) -> None:
        with pytest.raises(ValueError, match="shell metacharacters"):
            is_valid_import_string(value)

    def test_leading_dot_raises(self) -> None:
        with pytest.raises(ValueError, match="empty leading/trailing/interior part"):
            is_valid_import_string(".foo")

    def test_trailing_dot_raises(self) -> None:
        with pytest.raises(ValueError, match="empty leading/trailing/interior part"):
            is_valid_import_string("foo.")

    def test_consecutive_dots_raises(self) -> None:
        with pytest.raises(ValueError, match="empty leading/trailing/interior part"):
            is_valid_import_string("foo..bar")

    def test_numeric_prefix_part_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid Python identifiers"):
            is_valid_import_string("foo.123abc")

    def test_python_keyword_part_raises(self) -> None:
        with pytest.raises(ValueError, match="reserved keywords"):
            is_valid_import_string("foo.class")

    def test_keyword_in_first_part_raises(self) -> None:
        with pytest.raises(ValueError, match="reserved keywords"):
            is_valid_import_string("class.foo")

    # Bad-parts reporting — message names every offender, not just the first
    def test_error_message_names_all_bad_parts(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            is_valid_import_string("foo.123.class")
        # Both bad parts surface in the message
        assert "123" in str(exc_info.value)
        assert "class" in str(exc_info.value)

    # Round-trip with resolve_candidate_paths — every input rejected by the
    # predicate produces [] from the resolver; every input admitted by the
    # predicate produces a non-empty candidate list (when import_root exists)
    @pytest.mark.parametrize(
        "bad_value",
        [
            "",
            ".foo",
            "foo.",
            "foo..bar",
            "foo/bar",
            "foo\\bar",
            "foo;bar",
            "foo.123abc",
            "foo.class",
        ],
    )
    def test_resolver_returns_empty_for_predicate_rejections(
        self, bad_value: str, tmp_path: Path
    ) -> None:
        """Per the shared-predicate contract: resolver MUST return [] for every
        string the predicate rejects (caller-side `try/except ValueError`)."""
        # Sanity: predicate rejects
        with pytest.raises(ValueError):
            is_valid_import_string(bad_value)
        # Resolver returns empty
        assert resolve_candidate_paths(bad_value, tmp_path) == []

    @pytest.mark.parametrize(
        "good_value",
        ["foo", "foo.bar", "a.b.c.d", "foo._private", "foo.__init__"],
    )
    def test_resolver_returns_candidates_for_predicate_admits(
        self, good_value: str, tmp_path: Path
    ) -> None:
        """Per the shared-predicate contract: resolver MUST return non-empty
        candidate list for every string the predicate admits (when import_root
        is a real existing directory)."""
        # Sanity: predicate admits (returns the normalized form)
        normalized = is_valid_import_string(good_value)
        assert normalized
        # Resolver returns the two-candidate list
        assert len(resolve_candidate_paths(good_value, tmp_path)) == 2
