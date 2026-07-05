"""OBSERVED-tier query registry surface + digest behavior (Cost Lever 3).

Pins the contracts the deterministic OBSERVED producer (a later increment)
depends on, plus the cache-key digest fold that the round-3 review of
specs/2026-06-14-observed-query-library-v1.md added:

- Every OBSERVED query carries metadata (finding_type in SEVERITY_POLICY,
  a query_class, non-empty title/description).
- V1 is default-deny: every query is SIGNAL_ONLY (zero SKIP_SAFE seeds).
- OBSERVED ids resolve via get_query_source/match but are a SEPARATE
  surface from REGISTERED_QUERY_IDS (the structural LLM-citation set).
- `_registry_digest` folds every non-excluded ObservedQuery field (today
  language/class/finding_type/title/description/binding, derived from the
  model per FUP-181), so a metadata edit — not just a .scm body edit —
  invalidates the analyze cache.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from outrider.ast_facts.errors import UnknownQueryMatchId
from outrider.policy.severity import SEVERITY_POLICY, FindingType
from outrider.queries import registry
from outrider.queries.observed import BindingRule, ObservedQuery, QueryClass


def test_observed_query_count() -> None:
    """The OBSERVED library has nineteen queries: eleven Python (8 from the
    v1 library + weak_crypto_broken_cipher + weak_crypto_ecb_mode, FUP-193
    step 1; + weak_asymmetric_key_size, the value-predicate slice) + eight
    JS/TS (the four-family catalog,
    specs/2026-07-03-js-ts-observed-query-catalog.md, plus the process-env
    TLS kill switch split into its own import-free query in the
    import-binding fold)."""
    assert len(registry.OBSERVED_QUERY_IDS) == 19
    assert set(registry.OBSERVED_QUERIES) == set(registry.OBSERVED_QUERY_IDS)
    by_language = {"python": 0, "javascript": 0}
    for oq in registry.OBSERVED_QUERIES.values():
        by_language[oq.language] += 1
    assert by_language == {"python": 11, "javascript": 8}


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
    """match() resolves every OBSERVED id (loaded + compiled) under EVERY
    grammar its language compiles for — a JS/TS catalog query must resolve
    under javascript, typescript, AND tsx."""
    for oq in registry.OBSERVED_QUERIES.values():
        for grammar in registry._GRAMMARS_BY_QUERY_LANGUAGE[oq.language]:
            # Empty source: registered query, zero matches — must NOT raise.
            assert registry.match(oq.query_match_id, b"", grammar=grammar) == ()


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


def test_digest_excluded_fields_pinned() -> None:
    """FUP-181: the digest derives its OBSERVED fold from `ObservedQuery`'s model
    fields minus an explicit exclusion set, so a NEW output/routing field enters
    the cache key by DEFAULT. Pin the exclusion set (growing it removes a field
    from the cache key — a conscious cache-identity decision) and assert it names
    only real, non-output fields. The folds-metadata test above proves the
    non-excluded fields actually move the digest."""
    from outrider.queries.observed import ObservedQuery

    excluded = registry._DIGEST_EXCLUDED_OBSERVED_FIELDS
    model_fields = set(ObservedQuery.model_fields)
    assert excluded == {"query_match_id", "filename"}, (
        "digest exclusion set changed — adding a field here drops it from the "
        "cache key; confirm it truly affects neither emitted output nor routing"
    )
    assert excluded <= model_fields, (
        f"excluded names not on ObservedQuery: {excluded - model_fields}"
    )


def test_digest_folds_every_non_excluded_field() -> None:
    """FUP-181 robustness: mutating ANY non-excluded ObservedQuery field moves the
    digest — proves the model-derived fold covers each field, not just a hardcoded
    subset, so a future field can't silently miss the cache key."""
    bodies = dict(registry._QUERY_BODIES)
    base = registry._registry_digest(bodies, registry._OBSERVED_QUERIES, registry.VALUE_PREDICATES)
    oid = next(iter(registry._OBSERVED_QUERIES))
    orig = registry._OBSERVED_QUERIES[oid]
    folded_fields = set(registry._OBSERVED_QUERIES[oid].model_fields) - (
        registry._DIGEST_EXCLUDED_OBSERVED_FIELDS
    )
    other_ft = next(ft for ft in FindingType if ft is not orig.finding_type)
    # A distinct, type-valid mutation per folded field.
    mutations = {
        "finding_type": other_ft,
        "query_class": QueryClass.SKIP_SAFE,
        "language": "javascript" if orig.language == "python" else "python",
        "title": orig.title + " (edited)",
        "description": orig.description + " edited",
        # An admission-affecting rule change MUST re-key the cache: a match
        # admitted under one binding rule may be dropped under another.
        "binding": BindingRule(mode="module_presence", modules=("digest-probe",)),
        # Same admission-affecting property: adding/removing a guarded
        # global changes which matches the shadow guard denies.
        "shadow_guard": ("digest_probe_global",),
        # Eligibility flips which matches the module-scope arm admits —
        # a cached row from before the flip must not serve after it.
        "module_scope_eligible": not orig.module_scope_eligible,
    }
    assert folded_fields <= set(mutations), (
        f"new folded field(s) {folded_fields - set(mutations)} need a mutation case here"
    )
    for field in folded_fields:
        drifted = orig.model_copy(update={field: mutations[field]})
        mutated = {**registry._OBSERVED_QUERIES, oid: drifted}
        digest = registry._registry_digest(bodies, mutated, registry.VALUE_PREDICATES)
        assert digest != base, f"mutating non-excluded field {field!r} did not move the digest"


def test_observed_sweep_parses_source_once() -> None:
    """FUP-182: a clean file is parsed ONCE across its language's OBSERVED
    sweep, not once per query. match() memoizes the parse keyed by
    (source, grammar), so firing every same-language OBSERVED query against
    byte-identical source is a single tree-sitter parse (one cache miss) and
    the rest cache hits; a different source OR a different grammar is a fresh
    parse (the memo is per-(source, grammar), not a global parse-once)."""
    registry._parse_cached.cache_clear()
    python_ids = sorted(
        oq.query_match_id for oq in registry.OBSERVED_QUERIES.values() if oq.language == "python"
    )
    src = b"RSA.generate(1024)\nDES.new(key)\nyaml.load(data)\n"
    for qid in python_ids:
        registry.match(qid, src)
    info = registry._parse_cached.cache_info()
    assert info.misses == 1, f"OBSERVED sweep parsed {info.misses} times, want 1"
    assert info.hits == len(python_ids) - 1

    # A distinct source is a distinct parse — the memo is keyed by source bytes.
    registry.match(python_ids[0], b"DES.new(other)\n")
    assert registry._parse_cached.cache_info().misses == 2

    # A distinct GRAMMAR over byte-identical source is also a distinct parse:
    # the same bytes under two grammars are two different trees.
    js_ids = sorted(
        oq.query_match_id
        for oq in registry.OBSERVED_QUERIES.values()
        if oq.language == "javascript"
    )
    for qid in js_ids:
        registry.match(qid, src, grammar="javascript")
    info = registry._parse_cached.cache_info()
    assert info.misses == 3, "same bytes under a second grammar must be a fresh parse"
    assert info.hits == (len(python_ids) - 1) + (len(js_ids) - 1)
    registry._parse_cached.cache_clear()


def test_binding_none_javascript_queries_declare_a_shadow_guard() -> None:
    """Catalog-of-today contract (/code-review convergent find, angles B +
    altitude): a `binding=None` javascript query's ONLY lexical proof is its
    shadow guard — the anchor-shadow check inside `_binding_admits` never
    runs for it. Every current such entry text-constrains a global
    (`process.env`, `eval`/`Function`), so every one must declare the
    guarded names. A future binding=None query that genuinely needs no
    guard (pure-syntax pattern, no identifier constraint) updates this pin
    deliberately rather than skipping the guard by accident."""
    unguarded = sorted(
        oq.query_match_id
        for oq in registry.OBSERVED_QUERIES.values()
        if oq.language == "javascript" and oq.binding is None and not oq.shadow_guard
    )
    assert unguarded == [], (
        f"binding=None javascript queries without a shadow_guard: {unguarded} — "
        f"either guard their text-constrained globals or update this pin with "
        f"the rationale."
    )


def test_module_scope_eligibility_seeded_exactly_once() -> None:
    """The module-scope admission arm is opt-in and seeded on the one
    producer-pinned veto case (DECISIONS.md#062):
    the tls_env kill switch. Widening eligibility is an evidence-gated,
    deliberate act — a new eligible query updates this pin with its rationale,
    never rides in silently."""
    eligible = sorted(
        oq.query_match_id for oq in registry.OBSERVED_QUERIES.values() if oq.module_scope_eligible
    )
    assert eligible == ["javascript.tls_env_verify_disabled"]


def test_module_scope_eligible_requires_binding_none() -> None:
    """The SCHEMA floor rejects an eligible query carrying a BindingRule
    (DECISIONS.md#062: module-level admission would weaken an import-join
    proof) — a model_validator, so DIRECT construction cannot bypass it
    (revert-the-fold: deleting the validator fails this, unlike the prior
    registry-load loop whose deletion kept the suite green)."""
    base = next(iter(registry.OBSERVED_QUERIES.values()))
    with pytest.raises(ValidationError, match="module_scope_eligible requires binding=None"):
        ObservedQuery(
            **{
                **base.model_dump(exclude={"binding"}),
                "module_scope_eligible": True,
                "binding": BindingRule(mode="module_presence", modules=("probe",)),
            }
        )


def test_module_scope_eligible_rejects_skip_safe() -> None:
    """Eligibility may not combine with SKIP_SAFE promotion: module-arm
    matches are excluded from #049 skip coverage (diff-anchored proof, not
    scope-coverage-shaped), so a skip_safe eligible query would create
    coverage the skip contract forbids — rejected at the schema floor."""
    base = next(iter(registry.OBSERVED_QUERIES.values()))
    with pytest.raises(ValidationError, match="may not combine with SKIP_SAFE"):
        ObservedQuery(
            **{
                **base.model_dump(exclude={"binding"}),
                "binding": None,
                "module_scope_eligible": True,
                "query_class": QueryClass.SKIP_SAFE,
            }
        )
