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

from outrider.queries import registry

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
    MUST NOT fire against current OBSERVED admission (a deprecated query's
    semantics by definition no longer hold against current source). The
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
    — model OBSERVED claims on JS/TS reject by registration), and a
    catalog-less language (None) selects empty."""
    assert registry.structural_query_ids_for("python") == _EXPECTED_PYTHON_QUERY_IDS
    assert registry.structural_query_ids_for("javascript") == frozenset()
    assert registry.structural_query_ids_for(None) == frozenset()


def test_registered_query_ids_callers_get_consistent_set() -> None:
    """`REGISTERED_QUERY_IDS` is the same object across imports (module
    constant, not factory). Defends against a future refactor that
    wraps it in a property or function — the analyze node body holds a
    direct reference and assumes identity-stability per call."""
    first = registry.REGISTERED_QUERY_IDS
    second = registry.REGISTERED_QUERY_IDS
    assert first is second
