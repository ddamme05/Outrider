# Per DECISIONS.md#055 + specs/2026-06-21-cross-type-subsumption.md.
"""`verify_subsumption_wellformed()` fails loud on a malformed `SUBSUMES` map.

Monkeypatches the module attribute the guard reads, so each case exercises the
real guard against a deliberately-broken relation (the import-time floor that
fires even when `git commit --no-verify` bypasses CI)."""

from __future__ import annotations

from types import MappingProxyType

import pytest

import outrider.policy.subsumption as subsumption
from outrider.policy.severity import FindingType


def test_self_edge_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subsumption,
        "SUBSUMES",
        MappingProxyType({FindingType.WEAK_CRYPTO: frozenset({FindingType.WEAK_CRYPTO})}),
    )
    with pytest.raises(AssertionError, match="irreflexivity"):
        subsumption.verify_subsumption_wellformed()


def test_severity_lowering_edge_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # OPEN_REDIRECT (MEDIUM) "subsuming" SQL_INJECTION (CRITICAL) would let a
    # CRITICAL be masked by a less-severe finding — a monotonicity violation.
    monkeypatch.setattr(
        subsumption,
        "SUBSUMES",
        MappingProxyType({FindingType.OPEN_REDIRECT: frozenset({FindingType.SQL_INJECTION})}),
    )
    with pytest.raises(AssertionError, match="severity-monotonicity"):
        subsumption.verify_subsumption_wellformed()


def test_two_cycle_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ⊐ B and B ⊐ A makes the winner undefined. Both edges are severity-equal
    # (CRITICAL) so the cycle check fires before the monotonicity check.
    monkeypatch.setattr(
        subsumption,
        "SUBSUMES",
        MappingProxyType(
            {
                FindingType.SQL_INJECTION: frozenset({FindingType.AUTH_BYPASS}),
                FindingType.AUTH_BYPASS: frozenset({FindingType.SQL_INJECTION}),
            }
        ),
    )
    with pytest.raises(AssertionError, match="cycle"):
        subsumption.verify_subsumption_wellformed()


def test_chain_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ⊐ B ⊐ C: weak_password_hash ⊐ weak_crypto ⊐ open_redirect is a
    # severity-monotone CHAIN (would pass a 2-cycle-only check). V1 is single-hop,
    # so a subsumed type that is also a subsumer must be rejected.
    monkeypatch.setattr(
        subsumption,
        "SUBSUMES",
        MappingProxyType(
            {
                FindingType.WEAK_PASSWORD_HASH: frozenset({FindingType.WEAK_CRYPTO}),
                FindingType.WEAK_CRYPTO: frozenset({FindingType.OPEN_REDIRECT}),
            }
        ),
    )
    with pytest.raises(AssertionError, match="single-hop"):
        subsumption.verify_subsumption_wellformed()


def test_non_finding_type_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # A key that is not a real FindingType (e.g. a renamed/removed enum value
    # left dangling) must fail enum-membership validation.
    monkeypatch.setattr(
        subsumption,
        "SUBSUMES",
        MappingProxyType({"not_a_finding_type": frozenset({FindingType.WEAK_CRYPTO})}),
    )
    with pytest.raises(AssertionError, match="not a FindingType"):
        subsumption.verify_subsumption_wellformed()
