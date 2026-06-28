"""Unit tests for the OBSERVED value-predicate surface (FUP-193).

The value-predicate runs after the structural match to filter a query's matches
by a captured literal's value (a numeric comparison tree-sitter's native query
predicates cannot express). These tests pin the threshold semantics, the
literal-parse bases, the conservative drop on a non-literal, the loud failure on
a wiring mismatch, and the cache-key digest sensitivity to a parameter change.
"""

from __future__ import annotations

import pytest

from outrider.ast_facts.models import QueryCaptureSpan, QueryMatchSpan
from outrider.queries import registry
from outrider.queries.value_predicates import (
    VALUE_PREDICATE_CONTRACT_VERSION,
    VALUE_PREDICATES,
    ValuePredicate,
    _evaluate_weak_asymmetric_key_size,
)


def _match_with_keysize(literal: str) -> tuple[QueryMatchSpan, bytes]:
    """A QueryMatchSpan whose `_keysize` capture spans `literal` in the source."""
    src = f"RSA.generate({literal})".encode()
    start = src.index(literal.encode())
    end = start + len(literal.encode())
    cap = QueryCaptureSpan(name="_keysize", byte_start=start, byte_end=end)
    match = QueryMatchSpan(byte_start=start, byte_end=end, captures=(cap,))
    return match, src


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("1024", True),  # weak
        ("2047", True),  # just under the floor
        ("512", True),
        ("1_023", True),  # underscore literal
        ("0x400", True),  # hex 1024
        ("0o2000", True),  # octal 1024
        ("2048", False),  # floor is NOT flagged (strict `<`)
        ("4096", False),
        ("2_048", False),  # underscore 2048
        ("0x800", False),  # hex 2048
    ],
)
def test_threshold_and_parse_bases(literal: str, expected: bool) -> None:
    match, src = _match_with_keysize(literal)
    assert _evaluate_weak_asymmetric_key_size(match, src) is expected


def test_non_literal_keysize_drops_conservatively() -> None:
    """A capture text that does not parse as an int drops (False), never raises —
    an OBSERVED finding must not be claimed on a size it cannot evaluate."""
    match, src = _match_with_keysize("bits")
    assert _evaluate_weak_asymmetric_key_size(match, src) is False


def test_missing_keysize_capture_raises() -> None:
    """A match without the expected `_keysize` capture is a query/predicate wiring
    bug — it fails loud at first match rather than silently dropping findings."""
    cap = QueryCaptureSpan(name="_other", byte_start=0, byte_end=1)
    match = QueryMatchSpan(byte_start=0, byte_end=1, captures=(cap,))
    with pytest.raises(ValueError, match="_keysize"):
        _evaluate_weak_asymmetric_key_size(match, b"x")


def test_contract_token_encodes_threshold_and_version() -> None:
    """The token folded into the digest must carry the threshold AND the version,
    so a change to either invalidates cached analyze rows."""
    vp = VALUE_PREDICATES["python.weak_asymmetric_key_size"]
    assert "2048" in vp.contract_token
    assert VALUE_PREDICATE_CONTRACT_VERSION in vp.contract_token


def test_contract_token_change_moves_registry_digest() -> None:
    """A value-predicate parameter change (e.g. raising the threshold to 3072)
    must move QUERY_REGISTRY_DIGEST — the cache-invalidation contract."""
    base = registry._registry_digest(
        registry._QUERY_BODIES, registry._OBSERVED_QUERIES, registry.VALUE_PREDICATES
    )
    orig = registry.VALUE_PREDICATES["python.weak_asymmetric_key_size"]
    drifted = {
        **registry.VALUE_PREDICATES,
        "python.weak_asymmetric_key_size": ValuePredicate(
            evaluate=orig.evaluate,
            contract_token="weak_asymmetric_key_size:min_secure_bits=3072:v1",  # noqa: S106 (digest token, not a secret)
        ),
    }
    moved = registry._registry_digest(registry._QUERY_BODIES, registry._OBSERVED_QUERIES, drifted)
    assert moved != base, "a predicate token change must move the registry digest"


def test_every_value_predicate_keys_a_registered_observed_query() -> None:
    """A value-predicate keyed to a non-OBSERVED query id (unregistered, or a
    structural/deprecated id outside the OBSERVED producer's iteration) would
    silently no-op — the query over-fires with no value filter. The registry
    fails loud at module load; pin the OBSERVED-only invariant here too."""
    for qid in registry.VALUE_PREDICATES:
        assert qid in registry.OBSERVED_QUERY_IDS, (
            f"value-predicate {qid!r} does not key a registered OBSERVED query"
        )


def test_value_predicates_mapping_is_read_only() -> None:
    """MappingProxyType blocks in-process mutation (the OBSERVED_QUERIES
    precedent): the live match() filter cannot drift from the import-pinned
    QUERY_REGISTRY_DIGEST via a mutation of the predicate table."""
    with pytest.raises(TypeError):
        registry.VALUE_PREDICATES["python.x"] = None  # type: ignore[index]
