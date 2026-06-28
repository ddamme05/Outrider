"""OBSERVED-tier query registry surface + digest behavior (Cost Lever 3).

Pins the contracts the deterministic OBSERVED producer (a later increment)
depends on, plus the cache-key digest fold that the round-3 review of
specs/2026-06-14-observed-query-library-v1.md added:

- Every OBSERVED query carries metadata (finding_type in SEVERITY_POLICY,
  a query_class, non-empty title/description).
- V1 is default-deny: every query is SIGNAL_ONLY (zero SKIP_SAFE seeds).
- OBSERVED ids resolve via get_query_source/match but are a SEPARATE
  surface from REGISTERED_QUERY_IDS (the structural LLM-citation set).
- `_registry_digest` folds class/finding_type/title/description, so a
  metadata edit (not just a .scm body edit) invalidates the analyze cache.
"""

from __future__ import annotations

import pytest

from outrider.ast_facts.errors import UnknownQueryMatchId
from outrider.policy.severity import SEVERITY_POLICY, FindingType
from outrider.queries import registry
from outrider.queries.observed import QueryClass


def test_observed_query_count() -> None:
    """The OBSERVED seed library has eleven queries (8 from the v1 library +
    weak_crypto_broken_cipher + weak_crypto_ecb_mode, FUP-193 step 1; +
    weak_asymmetric_key_size, FUP-193 the value-predicate slice)."""
    assert len(registry.OBSERVED_QUERY_IDS) == 11
    assert set(registry.OBSERVED_QUERIES) == set(registry.OBSERVED_QUERY_IDS)


def test_observed_queries_default_deny_all_signal_only() -> None:
    """Default-deny per the spec: V1 promotes nothing to SKIP_SAFE."""
    for oq in registry.OBSERVED_QUERIES.values():
        assert oq.query_class is QueryClass.SIGNAL_ONLY, (
            f"{oq.query_match_id} is {oq.query_class}; V1 seeds zero SKIP_SAFE"
        )


def test_observed_finding_types_have_policy_severity() -> None:
    """Every finding_type maps in SEVERITY_POLICY so the producer can assign
    a deterministic severity (DECISIONS.md#001 + #048)."""
    for oq in registry.OBSERVED_QUERIES.values():
        assert oq.finding_type in SEVERITY_POLICY, (
            f"{oq.query_match_id} -> {oq.finding_type} has no SEVERITY_POLICY entry"
        )
        assert isinstance(oq.finding_type, FindingType)


def test_observed_queries_have_nonempty_static_text() -> None:
    """title/description are deterministic static text the producer writes
    into the finding (no model text)."""
    for oq in registry.OBSERVED_QUERIES.values():
        assert oq.title.strip()
        assert oq.description.strip()


def test_observed_ids_resolve_via_get_query_source() -> None:
    """OBSERVED ids resolve in the registry (so replay's get_query_source
    check passes for OBSERVED findings citing them)."""
    for qid in registry.OBSERVED_QUERY_IDS:
        assert registry.get_query_source(qid).strip()


def test_observed_filename_matches_id_resolution() -> None:
    """match() resolves every OBSERVED id (loaded + compiled)."""
    for qid in registry.OBSERVED_QUERY_IDS:
        # Empty source: registered query, zero matches — must NOT raise.
        assert registry.match(qid, b"") == ()


def test_observed_is_separate_from_structural_registered_set() -> None:
    """OBSERVED ids are NOT in REGISTERED_QUERY_IDS (the structural
    LLM-citation admission set); the two query KINDS stay distinct."""
    assert registry.OBSERVED_QUERY_IDS.isdisjoint(registry.REGISTERED_QUERY_IDS)


def test_observed_queries_mapping_is_read_only() -> None:
    """MappingProxyType blocks runtime mutation (defense-in-depth)."""
    with pytest.raises(TypeError):
        registry.OBSERVED_QUERIES["python.x"] = None  # type: ignore[index]


def test_unknown_observed_id_raises() -> None:
    with pytest.raises(UnknownQueryMatchId):
        registry.get_query_source("python.not_a_real_observed_query")


def test_digest_folds_observed_metadata() -> None:
    """A change to an OBSERVED query's class / finding_type / title /
    description changes the digest even with the .scm body unchanged — the
    round-3 cache-identity guard. Without this, a metadata edit could serve
    a stale cached analyze outcome."""
    bodies = dict(registry._QUERY_BODIES)
    base = registry._registry_digest(bodies, registry._OBSERVED_QUERIES, registry.VALUE_PREDICATES)

    oid = next(iter(registry._OBSERVED_QUERIES))
    orig = registry._OBSERVED_QUERIES[oid]
    other_ft = next(ft for ft in FindingType if ft is not orig.finding_type)

    drifts = (
        orig.model_copy(update={"title": orig.title + " (edited)"}),
        orig.model_copy(update={"description": orig.description + " edited"}),
        orig.model_copy(update={"query_class": QueryClass.SKIP_SAFE}),
        orig.model_copy(update={"finding_type": other_ft}),
    )
    for drifted in drifts:
        mutated = {**registry._OBSERVED_QUERIES, oid: drifted}
        assert registry._registry_digest(bodies, mutated, registry.VALUE_PREDICATES) != base, (
            "digest did not change when OBSERVED metadata changed; stale-cache risk"
        )


def test_digest_stable_under_no_change() -> None:
    """Recomputing the digest over the same inputs is deterministic."""
    a = registry._registry_digest(
        dict(registry._QUERY_BODIES), registry._OBSERVED_QUERIES, registry.VALUE_PREDICATES
    )
    b = registry._registry_digest(
        dict(registry._QUERY_BODIES), registry._OBSERVED_QUERIES, registry.VALUE_PREDICATES
    )
    assert a == b == registry.QUERY_REGISTRY_DIGEST
