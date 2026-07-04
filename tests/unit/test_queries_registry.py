"""Unit tests for `queries.registry` public surface.

Pins the contract that `REGISTERED_QUERY_IDS` is the all-languages union
of structural (model-citable) query ids, from which the analyze node's
per-file OBSERVED-admission set is selected via
`structural_query_ids_for(language)` — every claim cross-referenced
against that per-language set at `analyze_parser.py`'s producer-admission
step. A drift in construction (e.g., including deprecated ids, missing a
freshly-added query, or a language selecting another language's ids)
would silently shift which `query_match_id` claims the parser accepts;
pinning the set construction catches that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from outrider.queries import registry

if TYPE_CHECKING:
    from outrider.queries.observed import QueryLanguage

_EXPECTED_PYTHON_QUERY_IDS = frozenset(
    {
        "python.function_definition",
        "python.class_definition",
        "python.import_statement",
        "python.import_from_statement",
    }
)


def test_registered_query_ids_is_frozenset() -> None:
    """Hash-stable contract: the public surface is `frozenset[str]`,
    immutable, hashable, set-membership testable. A `set` mutable
    re-export would let a caller `.add(...)` a deprecated id back into
    OBSERVED admission at runtime."""
    assert isinstance(registry.REGISTERED_QUERY_IDS, frozenset)


def test_registered_query_ids_matches_current_query_id_map() -> None:
    """`REGISTERED_QUERY_IDS` MUST equal the keys of the live
    `_QUERY_ID_TO_FILENAME` mapping — they are constructed in lockstep
    by the registry's module-load step. A future query addition that
    extends the filename map but not the public surface would silently
    leave the OBSERVED-admission set stale."""
    assert frozenset(registry._QUERY_ID_TO_FILENAME.keys()) == registry.REGISTERED_QUERY_IDS  # noqa: SLF001


def test_registered_query_ids_excludes_deprecated() -> None:
    """Deprecated query ids exist for REPLAY of historical reviews; they
    MUST NOT fire against current OBSERVED admission (a ledger id's claim
    was RETIRED per DECISIONS.md#061 — it is no longer produced against
    current source; claim-preserving edits never enter the ledger). The
    construction `frozenset(_QUERY_ID_TO_FILENAME)` excludes the
    `_DEPRECATED_QUERY_ID_TO_BODY` mapping by design — pin that
    invariant so a future refactor that unions them is rejected loud."""
    assert registry.REGISTERED_QUERY_IDS.isdisjoint(
        registry._DEPRECATED_QUERY_ID_TO_BODY.keys()  # noqa: SLF001
    )


def test_registered_query_ids_pins_v1_python_set() -> None:
    """V1 ships exactly four Python structural queries and — deliberately —
    ZERO javascript ones (the JS/TS catalog is OBSERVED-only, so the union
    equals the python set); pin the membership so a silent removal (e.g.,
    during a query-refactor that drops `python.import_statement` and
    forgets to re-add it) surfaces here rather than as missed OBSERVED
    admissions at runtime."""
    assert registry.REGISTERED_QUERY_IDS == _EXPECTED_PYTHON_QUERY_IDS


def test_structural_query_ids_select_per_language() -> None:
    """The per-file admission selector: python selects the four structural
    ids, javascript selects the EMPTY set (no structural queries registered
    — model OBSERVED claims on JS/TS reject by registration), a
    catalog-less language (None) selects empty, and a language missing from
    the structural table entirely (a future catalog language whose entry
    hasn't landed) selects empty rather than raising mid-review — the
    fail-safe direction the registry module docstring pins."""
    assert registry.structural_query_ids_for("python") == _EXPECTED_PYTHON_QUERY_IDS
    assert registry.structural_query_ids_for("javascript") == frozenset()
    assert registry.structural_query_ids_for(None) == frozenset()
    future_language = cast("QueryLanguage", "go")
    assert registry.structural_query_ids_for(future_language) == frozenset()


def test_registered_query_ids_callers_get_consistent_set() -> None:
    """`REGISTERED_QUERY_IDS` is the same object across imports (module
    constant, not factory). Defends against a future refactor that
    wraps it in a property or function — the analyze node body holds a
    direct reference and assumes identity-stability per call."""
    first = registry.REGISTERED_QUERY_IDS
    second = registry.REGISTERED_QUERY_IDS
    assert first is second


# ---------------------------------------------------------------------------
# Anchor-capture protocol enforcement (`_validate_anchor_captures`): an
# `anchor_import` query whose `.scm` captures neither `_fn` nor `_recv`
# would be 100% default-denied by the producer, silently — the registry
# rejects it at import instead.
# ---------------------------------------------------------------------------


def test_anchor_capture_typo_rejected_at_import() -> None:
    """A typo'd anchor capture (`@_fun`) leaves the pattern with no anchor;
    `_validate_anchor_captures` raises instead of letting `_binding_admits`
    silently suppress every match of the query."""
    import pytest

    body = '(call_expression function: (identifier) @_fun (#eq? @_fun "exec")) @x'
    query = registry._compile_and_validate("javascript.probe", body, grammar="javascript")  # noqa: SLF001
    with pytest.raises(ValueError, match="anchor_import.*neither '_fn' nor '_recv'"):
        registry._validate_anchor_captures("javascript.probe", query, grammar="javascript")  # noqa: SLF001


def test_anchor_capture_check_is_per_pattern() -> None:
    """A multi-pattern file where ONE pattern lacks the anchor still fails —
    file-level capture presence is not enough (that pattern's matches would
    be default-denied while its siblings work, the silent-partial failure)."""
    import pytest

    body = (
        "(call_expression function: (identifier) @_fn) @x\n"
        "(call_expression function: (member_expression object: (identifier) @_obj)) @x\n"
    )
    query = registry._compile_and_validate("javascript.probe", body, grammar="javascript")  # noqa: SLF001
    with pytest.raises(ValueError, match="pattern 1"):
        registry._validate_anchor_captures("javascript.probe", query, grammar="javascript")  # noqa: SLF001


def test_live_anchor_import_queries_satisfy_the_protocol() -> None:
    """Every registered anchor_import query passes the validator under every
    grammar it compiles for — the import-time wiring's positive control (and
    the proof the live catalog's capture names are protocol-conformant)."""
    checked = 0
    for query_id, observed in registry._OBSERVED_QUERIES.items():  # noqa: SLF001
        if observed.binding is None or observed.binding.mode != "anchor_import":
            continue
        for grammar, query in registry._COMPILED_QUERIES[query_id].items():  # noqa: SLF001
            registry._validate_anchor_captures(query_id, query, grammar=grammar)  # noqa: SLF001
            checked += 1
    assert checked, "the live catalog must register at least one anchor_import query"


def test_shadow_guard_requires_a_guard_position_capture() -> None:
    """A shadow_guard query must capture its global at a guard-POSITION
    (`GUARD_POSITION_CAPTURES`); the producer only tests guarded names
    there, so a query pinning its global under another capture name would
    have a silently-inert guard (/code-review find). The registry rejects
    it at load, mirroring `_validate_anchor_captures`."""
    import pytest

    from outrider.queries.observed import GUARD_POSITION_CAPTURES

    body = '(call_expression function: (identifier) @_target (#eq? @_target "x")) @m'
    query = registry._compile_and_validate("javascript.probe", body, grammar="javascript")  # noqa: SLF001
    with pytest.raises(ValueError, match="no guard-position identifier"):
        registry._validate_guard_position_captures("javascript.probe", query, grammar="javascript")  # noqa: SLF001
    # Every live shadow_guard query passes under each of its grammars.
    checked = 0
    for query_id, observed in registry._OBSERVED_QUERIES.items():  # noqa: SLF001
        if not observed.shadow_guard:
            continue
        assert set(observed.shadow_guard)  # sanity
        for grammar, query in registry._COMPILED_QUERIES[query_id].items():  # noqa: SLF001
            registry._validate_guard_position_captures(query_id, query, grammar=grammar)  # noqa: SLF001
            checked += 1
    assert checked, "expected at least one live shadow_guard query"
    assert GUARD_POSITION_CAPTURES  # imported symbol used


def test_guard_position_capture_check_is_per_pattern() -> None:
    """A multi-pattern file where ONE pattern lacks a guard-position capture
    still fails — query-wide capture presence is not enough (the mis-captured
    sibling's matches would carry no guard-position capture, leaving
    `_guarded_global_shadowed` silently inert for exactly that pattern while
    the well-captured sibling masks the hole). Mirrors
    `test_anchor_capture_check_is_per_pattern`."""
    import pytest

    body = (
        '(call_expression function: (identifier) @_fn (#eq? @_fn "eval")) @m\n'
        '(new_expression constructor: (identifier) @_target (#eq? @_target "Function")) @m\n'
    )
    query = registry._compile_and_validate("javascript.probe", body, grammar="javascript")  # noqa: SLF001
    with pytest.raises(ValueError, match="pattern 1"):
        registry._validate_guard_position_captures("javascript.probe", query, grammar="javascript")  # noqa: SLF001
