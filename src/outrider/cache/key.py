# Per specs/2026-06-11-file-hash-analyze-cache.md — the analyze-cache key.
# Per DECISIONS.md#056: the host-identity triad (profile_id, reasoning_enabled,
# profile_contract_digest) joins the key, splitting the analyze cache by host.
"""Cache-key composition for the file-hash analyze cache (lever #8).

One pure function: every input that could change a per-file analyze
outcome becomes a length-prefixed component of one SHA-256. The rendered
prompt rides as `llm/base.py::_canonical_prompt_hash` output (one
recipe, two consumers — it must never fork from `LLMCallEvent.prompt_hash`);
the explicit components cover what prompt bytes can't see — including the
parameterized-call veto's per-file scan facts (FUP-171), which decide
admission from whole-file bytes the rendered prompt never shows — and
`(installation_id, repo_id)` is the tenant boundary (installation is the
root — `repo_id` is only unique per installation).

Pure and import-light by design: callers pass the version values
(`TRIVIAL_FILTER_VERSION` loads tree-sitter; this module must not).
Length-prefixing every component makes boundaries unambiguous — adjacent
components can never collide by shifting bytes across a delimiter
(`("AB","C")` vs `("A","BC")`), which matters because several components
derive from attacker-influenced content.
"""

from __future__ import annotations

import hashlib
from typing import Final

from outrider.llm.base import _canonical_prompt_hash

# The analyze cache-key RECIPE version (DECISIONS.md#056). Distinct from
# ANALYZE_PARSER_VERSION, which versions admitted-findings SEMANTICS: this
# versions the cache-key RECIPE STRUCTURE — the component set, their order, and
# the length-prefix framing. Bump on ANY recipe change (add/remove/reorder a
# component, change the framing). v1 was the implicit original recipe (no
# constant existed); v2 is the host-identity re-key (#056 folded the triad into
# the key); v3 folds `from_import_map_digest` — candidate correction (#024
# from-import amendment) makes cached trace_candidates depend on module-level
# imports the rendered prompt doesn't carry; v4 folds
# `import_bindings_digest` — import-binding OBSERVED admission
# consumes ALL of the file's imports (module + local
# names), which the from-import map deliberately excludes (Python-shaped
# validation over `from`-kind refs only); v5 folds
# `lexical_bindings_digest` — the shadowing guard reads local-binding
# visibility spans that can live in enclosing-but-not-included scopes the
# prompt never shows (the import digest also widened to carry the
# value-import marker in the same arc); v6 folds `module_admission_digest`
# (DECISIONS.md#062) — the module-scope
# admission arm consumes head-side added-line ranges, the module-level
# bytes they cover, and every parsed scope span, all outside prompt
# bytes. The recipe change self-invalidates
# old rows on its own, but the explicit version is the legible, replay-durable
# marker #056 mandates ("the analyze cache-keyed version bumps") and gives
# future non-parser recipe changes a home without overloading
# ANALYZE_PARSER_VERSION's admission-only scope.
ANALYZE_CACHE_KEY_VERSION: Final = "analyze-cache-key-v6"


def compute_analyze_cache_key(
    *,
    system_prompt: str,
    user_prompt: str,
    installation_id: int,
    repo_id: int,
    model: str,
    prompt_template_version: str,
    trivial_filter_version: str,
    query_registry_digest: str,
    active_policy_version: str,
    analyze_parser_version: str,
    response_format_digest: str,
    parameterized_call_scan_digest: str,
    observed_producer_version: str,
    subsumes_digest: str,
    from_import_map_digest: str,
    import_bindings_digest: str,
    lexical_bindings_digest: str,
    module_admission_digest: str,
    profile_id: str | None,
    reasoning_enabled: bool | None,
    profile_contract_digest: str | None,
) -> str:
    """The analyze-cache key: length-prefixed fields — the recipe
    version (`ANALYZE_CACHE_KEY_VERSION`) first, then the canonical prompt
    digest, then the explicit scope/version/identity components — as
    one SHA-256 hex digest. `from_import_map_digest`
    (#024 from-import correction) pins the analyzed file's from-import
    name→module map: corrected sibling candidates depend on module-level
    imports the rendered prompt doesn't carry (scope bodies + hunks only),
    so two reviews with byte-identical prompts but different from-imports
    admit different trace_candidates and must never share an entry.
    `import_bindings_digest` pins the import-binding admission step's
    per-file input — the `(module, is_value_import, names)` view of ALL
    the file's imports, which `_binding_admits` joins name-anchored
    OBSERVED matches against; the from-import map excludes most JS/TS forms
    (`node:`-prefixed / hyphenated specifiers, "direct"-kind whole-module /
    namespace / side-effect imports), so without this component two reviews
    with byte-identical prompts but different imports would share an entry
    and serve a stale admitted-finding set.
    `lexical_bindings_digest` pins the shadowing guard's per-file input —
    the `(name, visibility span)` view of the file's local bindings, which
    can live in enclosing-but-not-included scopes the prompt never shows;
    without it a shadowing edit outside the shown scopes would serve the
    pre-edit admitted-finding set.
    `module_admission_digest`
    (DECISIONS.md#062) pins the module-scope
    arm's per-file input: the head-side added-line byte ranges, the
    module-level bytes they cover, and every parsed scope span (the
    disjointness predicate's input) — all outside prompt bytes, so two
    reviews with byte-identical prompts but a different module-level
    change (or a different scope layout around it) must never share an
    entry.
    `subsumes_digest` (DECISIONS.md#055) pins the
    `SUBSUMES` cross-type relation's CONTENT: cross-type subsumption drops an
    admitted OBSERVED finding under a same-span JUDGED subsumer, so a relation
    edge edit changes the admitted finding set without touching the prompt,
    registry digest, or parser version — it must invalidate entries.
    `observed_producer_version` (Cost Lever 3) pins the
    deterministic OBSERVED producer's ADMISSION logic (per-language query-set
    selection, scope-containment, test-file suppression, import-binding
    admission, zero-width skip, byte→line mapping) — a change there
    alters the cached finding set without touching the prompt, the registry
    digest, or the parser version, so it must invalidate entries.
    `response_format_digest` (FUP-096) pins the
    request format: constrained-decoding and free-form calls are
    different output populations for identical prompt bytes, so they
    must never share an entry (pass a fixed sentinel such as
    `"none"` only if a caller genuinely has no format concept — analyze
    always passes the real digest). `parameterized_call_scan_digest`
    (FUP-171) pins the parameterized-call veto's per-file input: two
    reviews with byte-identical prompts but a syntax error in an
    out-of-scope region admit different finding sets (the veto disables
    on any whole-file parse error), so they must never share an entry.

    The host-identity triad — `profile_id`, `reasoning_enabled`,
    `profile_contract_digest` (DECISIONS.md#056) — splits the cache by
    provider host: identical prompt bytes sent to different hosts
    (Baseten-GLM vs Anthropic), or to one host with reasoning on vs off,
    are different output populations and must never share an entry. All
    three `None` is an UNQUALIFIED (pre-#056) caller; each folds an empty
    component, which never collides with a real host (`host_id`/digest are
    non-empty, and `reasoning_enabled` renders `true`/`false`).

    `ANALYZE_CACHE_KEY_VERSION` leads as the recipe-structure version: any
    change to the component set, their order, or the framing bumps it (the
    explicit, replay-durable marker #056 requires). Component order is itself
    part of the recipe — changing it is a cache-wide invalidation and must be
    deliberate. `active_policy_version` is the THREADED write-time value analyze
    stamps findings with (never the module constant read here); same for every
    version argument.
    """
    # Host-identity triad (DECISIONS.md#056) are peers — all-present (a QUALIFIED
    # host call) or all-None (an UNQUALIFIED pre-#056 caller). A partial triad is
    # incoherent: no valid #056 audit event can represent it (LLMResponse and the
    # three completion events enforce the same all-or-none envelope), so folding
    # one here would mint a cache key no event matches. build_graph rejects partials
    # upstream; this guards direct callers of the exported helper, keeping the
    # invariant total across every triad boundary.
    _triad = (profile_id, reasoning_enabled, profile_contract_digest)
    if any(v is not None for v in _triad) and not all(v is not None for v in _triad):
        raise ValueError(
            "host-identity triad (DECISIONS.md#056) is peers — all-present or all-None; "
            f"got a partial set: profile_id={profile_id!r}, "
            f"reasoning_enabled={reasoning_enabled!r}, "
            f"profile_contract_digest={'<set>' if profile_contract_digest is not None else None}"
        )
    prompt_digest = _canonical_prompt_hash(system_prompt=system_prompt, user_prompt=user_prompt)
    # Triad fold: None => UNQUALIFIED, folded as an empty component; reasoning
    # renders true/false. Never collides with a real host (host_id/digest are
    # non-empty).
    reasoning_component = (
        "" if reasoning_enabled is None else ("true" if reasoning_enabled else "false")
    )
    h = hashlib.sha256()
    for component in (
        ANALYZE_CACHE_KEY_VERSION,
        prompt_digest,
        str(installation_id),
        str(repo_id),
        model,
        prompt_template_version,
        trivial_filter_version,
        query_registry_digest,
        active_policy_version,
        analyze_parser_version,
        response_format_digest,
        parameterized_call_scan_digest,
        observed_producer_version,
        subsumes_digest,
        from_import_map_digest,
        import_bindings_digest,
        lexical_bindings_digest,
        module_admission_digest,
        profile_id if profile_id is not None else "",
        reasoning_component,
        profile_contract_digest if profile_contract_digest is not None else "",
    ):
        component_bytes = component.encode("utf-8")
        h.update(f"{len(component_bytes)}:".encode())
        h.update(component_bytes)
    return h.hexdigest()
