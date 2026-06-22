# Per DECISIONS.md#055 + specs/2026-06-21-cross-type-subsumption.md.
"""`SUBSUMES` relation: the shipped map is well-formed + `subsumes()` truth table."""

from __future__ import annotations

from outrider.policy.severity import SEVERITY_POLICY, FindingSeverity, FindingType
from outrider.policy.subsumption import (
    SUBSUMES,
    SUBSUMES_DIGEST,
    subsumes,
    verify_subsumption_wellformed,
)


def test_shipped_relation_is_wellformed() -> None:
    # The module-load guard already ran at import; calling it again is the
    # explicit assertion that the shipped map passes every check.
    verify_subsumption_wellformed()


def test_seed_edge_present() -> None:
    assert SUBSUMES[FindingType.WEAK_PASSWORD_HASH] == frozenset({FindingType.WEAK_CRYPTO})


def test_subsumes_truth_table() -> None:
    # The seed edge is directional: the more-specific password-hash subsumes the
    # broader weak-crypto, never the reverse.
    assert subsumes(FindingType.WEAK_PASSWORD_HASH, FindingType.WEAK_CRYPTO) is True
    assert subsumes(FindingType.WEAK_CRYPTO, FindingType.WEAK_PASSWORD_HASH) is False
    # No declared edge → no subsumption (unrelated types coexist).
    assert subsumes(FindingType.SQL_INJECTION, FindingType.WEAK_CRYPTO) is False
    assert subsumes(FindingType.WEAK_CRYPTO, FindingType.WEAK_CRYPTO) is False


def test_relation_is_irreflexive_and_single_hop() -> None:
    # No type subsumes itself, and no subsumed type is itself a subsumer (V1 has
    # no chains) — the guard enforces these, this pins them against the map.
    keys = set(SUBSUMES)
    for subsumer, subsumed_set in SUBSUMES.items():
        assert subsumer not in subsumed_set
        assert keys.isdisjoint(subsumed_set)


def test_relation_is_severity_monotone() -> None:
    rank = list(FindingSeverity)
    for subsumer, subsumed_set in SUBSUMES.items():
        for subsumed in subsumed_set:
            # Lower index = more severe; the subsumer must be >= as severe.
            assert rank.index(SEVERITY_POLICY[subsumer]) <= rank.index(SEVERITY_POLICY[subsumed])


def test_digest_is_stable_64_hex() -> None:
    import re

    assert re.fullmatch(r"[0-9a-f]{64}", SUBSUMES_DIGEST)
