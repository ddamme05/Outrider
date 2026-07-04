# Deterministic OBSERVED-tier finding producer per
# specs/2026-06-14-observed-query-library-v1.md (Cost Lever 3).
"""Run the OBSERVED security queries once, then turn the matches into
`ReviewFinding`s with NO model text.

`run_observed_matches` is the single deterministic query pass for a file: it
runs every OBSERVED query registered for the file's catalog language (via
`queries.registry.match` under the file's grammar — the registry owns both
the tree-sitter execution and the per-language selection; a catalog-less
language yields zero queries and the producer stays inert), applies the
admission rules (test-file suppression, scope containment, import-binding
admission, zero-width skip,
byte->line mapping), and returns plain `ObservedMatch` domain records —
`query_class` included. Every consumer reads
the SAME records, so there is one definition of "the OBSERVED query facts for
this file": `produce_observed_findings` builds findings from them, and the skip
routing (the routing increment) computes coverage from the `skip_safe` subset.

A match becomes a finding by a fixed mapping: `finding_type` / `severity` /
`dimension` are policy-set (never model-set), `title` / `description` are the
registry's static text, the byte envelope maps to source lines through
`coordinates.query_span_to_source_lines`, and `evidence` is the matched source
text. Construction goes through `ReviewFinding(...)` so `enforce_proof_boundary`
(OBSERVED ⇒ non-empty `query_match_id`) and `_verify_content_hash` validate at
the schema floor.

CLEAN-parse only: the caller gates on `degraded_mode` (no OBSERVED findings on a
degraded/failed parse). In the default/production config OBSERVED findings AUGMENT
the LLM pass and never skip it (the registry seeds zero `skip_safe` queries and
`analyze_observed_skip_enforced` defaults False). `compute_observed_skip_shadow`
records the per-file `would_skip` / `not_eligible` decision; the ENFORCED pre-LLM
skip that consumes it is wired in `analyze._process_one_file` behind that flag
(Step 3b-mechanism, `DECISIONS.md#049`), dormant until a query is promoted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, assert_never

from outrider.audit.events import (
    ObservedSkipChangedRegion,
    ObservedSkipCoveringMatch,
    ObservedSkipShadowEvent,
    compute_finding_content_hash,
)
from outrider.coordinates import (
    CoordinateError,
    changed_line_spans,
    query_span_to_source_lines,
)
from outrider.policy.canonical import compute_proposal_hash
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import lookup_severity
from outrider.queries import registry as query_registry
from outrider.queries.observed import QueryClass
from outrider.schemas import ReviewFinding

if TYPE_CHECKING:
    from uuid import UUID

    from unidiff import PatchedFile

    from outrider.ast_facts.models import ImportRef, QueryCaptureSpan, ScopeUnit
    from outrider.policy.severity import FindingType
    from outrider.queries.observed import BindingRule, QueryLanguage


# Version of the OBSERVED producer's ADMISSION logic — the rules in
# `run_observed_matches` that decide which matches are admitted: the
# per-language query-set selection, the scope-CONTAINMENT predicate, test-file
# suppression (`_is_test_file`), the import-binding step
# (`_binding_admits`), the zero-width-span skip, and the byte->line
# mapping. Folded into the analyze cache key
# (`cache.key.compute_analyze_cache_key`) so a change to these rules
# invalidates cached analyze outcomes written under the old rules — the same
# staleness guard `ANALYZE_PARSER_VERSION` gives the LLM parser. The query `.scm`
# bodies + their registry metadata are pinned SEPARATELY by
# `QUERY_REGISTRY_DIGEST`; this constant covers only the producer's admission
# logic. BUMP on any rule change.
# v2 (JS/TS OBSERVED catalog): the query set became per-language-selected and
# `_is_test_file` learned the JS/TS test conventions (`__tests__/`,
# `*.test.*`, `*.spec.*`).
# v3: the JS/TS conventions — the inner `.test.`/`.spec.` NAME markers and
# the `__tests__/` DIRECTORY marker — became language-scoped. Under v2 they
# applied to every language, silently suppressing OBSERVED findings on
# Python production files: `report.spec.py` (name marker),
# `pkg/__tests__/util.py` (directory marker).
# v4: import-binding admission (`_binding_admits`) — a name-anchored match
# is admitted only when its anchor identifier is bound by an import from the
# query's `BindingRule.modules` (or, mode="module_presence", when the file
# imports one of them). Under v3 any callee/receiver NAME matched — an
# `exec` helper imported from `./jobs` produced a CRITICAL OBSERVED finding.
# v5: binding-module matching is package-root aware — a subpath specifier
# (`require("mysql2/promise")`) satisfies a rule naming its package root.
# Under v4 the join was exact-string, silently dropping matches whose file
# imported the driver through the dominant subpath idiom.
OBSERVED_PRODUCER_VERSION: Final[str] = "observed-producer-v5"

# A query match envelope spans the whole matched construct (e.g. an entire
# `cursor.execute(f"...long SQL...")` call), so the matched source can exceed
# `ReviewFinding.evidence`'s `max_length=2000`. Truncate to that cap so a long but
# legitimate match yields a (truncated-evidence) finding rather than a
# ValidationError that would crash analyze for the file. Truncation only affects
# >2000-char matches (which previously crashed, so never reached the cache), so it
# needs no OBSERVED_PRODUCER_VERSION bump.
_EVIDENCE_MAX_CHARS: Final[int] = 2000


@dataclass(frozen=True, slots=True)
class ObservedMatch:
    """One admitted OBSERVED query match for a file — plain domain data shared by
    every consumer of the single query pass (`run_observed_matches`).

    Carries the registry facts (`query_match_id`, `query_class`, `finding_type`,
    static `title` / `description`), the matched `evidence` text, and the
    1-indexed inclusive HEAD source-line envelope. No tree-sitter object crosses
    this boundary — raw nodes stay in `queries/`; this is the domain record the
    agent layer consumes.
    """

    query_match_id: str
    query_class: QueryClass
    finding_type: FindingType
    title: str
    description: str
    evidence: str
    line_start: int
    line_end: int


def _is_test_file(file_path: str, language: QueryLanguage | None) -> bool:
    """A repo-relative path is test code if any directory component is `tests`/
    `test`, or the filename is `conftest.py` / `test_*` / `*_test.py`. For
    JS/TS files (`language == "javascript"`) two ecosystem conventions apply
    on top: a `__tests__` directory component and the tier marker as an inner
    dotted segment (`auth.test.ts`, `login.spec.js`). Those two are
    language-scoped because dotted segments are ordinary naming elsewhere —
    `report.spec.py` is a production Python file, not a test.

    Per `docs/spec.md` §11.2: a security pattern in test code (e.g. `eval()` in a
    test fixture) is NOT a production finding. The deterministic producer can't
    make the test-context judgment the LLM does, so it suppresses OBSERVED
    findings in test files structurally.
    """
    path = PurePosixPath(file_path)
    if any(part in ("tests", "test") for part in path.parts[:-1]):
        return True
    name = path.name
    if name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py"):
        return True
    if language != "javascript":
        return False
    if "__tests__" in path.parts[:-1]:
        return True
    inner_segments = name.split(".")[1:-1]
    return any(seg in ("test", "spec") for seg in inner_segments)


def _module_matches(specifier: str, modules: tuple[str, ...]) -> bool:
    """True when an import specifier resolves into one of the rule's
    packages: an exact match, or a subpath of one (`mysql2/promise`,
    `pg/lib/...`) — Node resolves `pkg/subpath` inside `pkg`, so the
    subpath import proves the same package presence. The prefix is
    `/`-delimited, so a lookalike package (`mysql2-mock`) never matches.
    """
    return any(specifier == m or specifier.startswith(m + "/") for m in modules)


def _binding_admits(
    binding: BindingRule | None,
    captures: tuple[QueryCaptureSpan, ...],
    content_bytes: bytes,
    import_refs: tuple[ImportRef, ...],
) -> bool:
    """Deterministic import-binding admission for a name-anchored match.

    `anchor_import`: the anchor identifier — the `@_recv` receiver capture
    when present, else the `@_fn` callee capture — must be a LOCAL name
    bound by an import whose `module` matches the rule's set (`ImportRef.names`
    carries local binding names for ESM named/default/namespace and CJS
    `require` forms alike). A match with neither capture is NOT admitted —
    default-deny, the proof-boundary direction. `module_presence`: the file
    must import at least one of the rule's modules. Module matching is
    package-root aware in both modes (`_module_matches`). `binding=None`
    admits on structure alone (globals / self-proving patterns).

    Known precision residuals (signal_only-acceptable; FUP-214): the join is
    file-level, not lexically scoped — a local declaration SHADOWING an
    imported name still admits — and `ImportRef` does not distinguish
    `import type` / `export … from` re-exports / side-effect imports from
    value imports, so those also satisfy the rule. Closing either needs
    adapter-layer work (scope analysis; a type-only/re-export marker), not a
    producer edit.
    """
    if binding is None:
        return True
    if binding.mode == "module_presence":
        return any(_module_matches(ref.module, binding.modules) for ref in import_refs)
    if binding.mode == "anchor_import":
        anchor: str | None = None
        for wanted_name in ("_recv", "_fn"):
            for c in captures:
                if c.name == wanted_name:
                    anchor = content_bytes[c.byte_start : c.byte_end].decode(
                        "utf-8", errors="replace"
                    )
                    break
            if anchor is not None:
                break
        if anchor is None:
            return False
        return any(
            _module_matches(ref.module, binding.modules) and anchor in ref.names
            for ref in import_refs
        )
    assert_never(binding.mode)


def import_bindings_digest(import_refs: tuple[ImportRef, ...]) -> str:
    """SHA-256 over the binding-relevant view of the file's imports — the
    analyze-cache-key component pinning `_binding_admits`'s per-file input
    (the `from_import_map_digest` / FUP-171 precedent).

    Import-binding admission decides which OBSERVED matches survive from
    module-level imports the rendered prompt does NOT carry (scope-unit
    bodies + hunks only), so two reviews with byte-identical prompts but
    different imports admit different finding sets and must never share a
    cache entry. `from_import_map_digest` cannot cover this: it folds only
    `from`-kind refs whose module survives the Python-shaped
    `is_valid_import_string` gate, which excludes exactly the JS/TS binding
    inputs (`node:`-prefixed and hyphenated specifiers; whole-module
    `require` / namespace / default / side-effect imports are kind
    "direct").

    Hashes what admission consumes and nothing else: the set of
    `(module, names)` pairs — order- and duplicate-insensitive (admission
    is `any(...)` over refs), with `import_kind` / `line` / `file_path`
    excluded because admission ignores them. Count headers + length
    prefixes make boundaries unambiguous (the cache-key framing rule);
    the empty tuple digests distinctly from any populated set.
    """
    entries = sorted({(ref.module, tuple(sorted(ref.names))) for ref in import_refs})
    h = hashlib.sha256()
    h.update(f"{len(entries)}:".encode())
    for module, names in entries:
        module_bytes = module.encode("utf-8")
        h.update(f"{len(module_bytes)}:".encode())
        h.update(module_bytes)
        h.update(f"{len(names)}:".encode())
        for name in names:
            name_bytes = name.encode("utf-8")
            h.update(f"{len(name_bytes)}:".encode())
            h.update(name_bytes)
    return h.hexdigest()


def run_observed_matches(
    *,
    file_path: str,
    head_content: str,
    included_scope_units: tuple[ScopeUnit, ...],
    import_refs: tuple[ImportRef, ...],
) -> tuple[ObservedMatch, ...]:
    """Run every OBSERVED query registered for `file_path`'s catalog language
    over `head_content` and return the admitted matches as `ObservedMatch`
    records — the single deterministic query pass every downstream consumer
    reads. A file whose language has no catalog (or no registered adapter)
    selects zero queries and returns empty — the producer is inert by
    registration, so no OBSERVED finding is constructible for it.

    A match is admitted only when it is CONTAINED in an included scope unit (the
    same scope discipline the LLM-citation admission uses) — a deterministic
    OBSERVED match must anchor fully inside code the review is examining, never
    straddling a scope boundary or landing in unchanged code outside the diff —
    AND when its query's `BindingRule` is satisfied against `import_refs`
    (`_binding_admits`): a name-anchored match must prove its anchor binds to
    the dangerous API, so an `exec` helper imported from `./jobs` is not a
    `child_process` sink. Test files admit nothing (spec §11.2). Zero-width
    envelopes and spans that fail the byte->line mapping are dropped. Returns a
    deterministically ordered tuple (by query id, then the registry's match
    sort) so any content-derived id downstream stays stable across replays.
    """
    # Per-language selection: the registry pairs the query set with the
    # grammar that parses this file's bytes. Both None/empty for a
    # catalog-less language — the loop below then never runs.
    language = query_registry.query_language_for_path(file_path)
    grammar = query_registry.grammar_for_path(file_path)
    if language is None or grammar is None:
        return ()
    # Test code is not a production security finding (spec §11.2). The
    # language scopes the JS/TS-only name conventions.
    if _is_test_file(file_path, language):
        return ()
    content_bytes = head_content.encode("utf-8")
    scope_ranges = tuple((su.byte_start, su.byte_end) for su in included_scope_units)
    matches: list[ObservedMatch] = []

    for query_id, observed in query_registry.observed_queries_for(language).items():
        for span in query_registry.match(query_id, content_bytes, grammar=grammar):
            # A degenerate (zero-width) envelope has no reviewable line range;
            # skip rather than feed query_span_to_source_lines a span it rejects.
            if span.byte_end <= span.byte_start:
                continue
            # Containment: the match must fall fully inside an included scope's
            # byte range (a call always nests within its enclosing scope).
            if not any(s <= span.byte_start and span.byte_end <= e for s, e in scope_ranges):
                continue
            # Import-binding: a name-anchored match must prove its anchor
            # binds to the dangerous API (the import-binding admission step).
            if not _binding_admits(observed.binding, span.captures, content_bytes, import_refs):
                continue
            try:
                line_start, line_end = query_span_to_source_lines(
                    byte_start=span.byte_start,
                    byte_end=span.byte_end,
                    head_content=head_content,
                )
            except CoordinateError:
                # Out-of-bounds / unlocatable span — cannot anchor a match.
                continue

            evidence = content_bytes[span.byte_start : span.byte_end].decode(
                "utf-8", errors="replace"
            )[:_EVIDENCE_MAX_CHARS]
            matches.append(
                ObservedMatch(
                    query_match_id=observed.query_match_id,
                    query_class=observed.query_class,
                    finding_type=observed.finding_type,
                    title=observed.title,
                    description=observed.description,
                    evidence=evidence,
                    line_start=line_start,
                    line_end=line_end,
                )
            )
    return tuple(matches)


def produce_observed_findings(
    matches: tuple[ObservedMatch, ...],
    *,
    file_path: str,
    review_id: UUID,
    installation_id: int,
    active_policy_version: str,
) -> tuple[ReviewFinding, ...]:
    """Build deterministic OBSERVED `ReviewFinding`s from the matches of
    `run_observed_matches`.

    EVERY OBSERVED match becomes a finding — both `signal_only` and `skip_safe`;
    the class affects skip ROUTING, not whether a finding is emitted. Findings
    preserve the matches' deterministic order so the round's content-derived id
    stays stable across replays.
    """
    findings: list[ReviewFinding] = []
    for m in matches:
        proposal_hash = compute_proposal_hash(
            source_file_path=file_path,
            finding_type=m.finding_type.value,
            evidence_tier=EvidenceTier.OBSERVED.value,
            query_match_id=m.query_match_id,
            trace_path=None,
            title=m.title,
            description=m.description,
            evidence=m.evidence,
            line_start=m.line_start,
            line_end=m.line_end,
        )
        findings.append(
            ReviewFinding(
                review_id=review_id,
                installation_id=installation_id,
                policy_version=active_policy_version,
                finding_type=m.finding_type,
                dimension=lookup_dimension(m.finding_type),
                severity=lookup_severity(m.finding_type),
                evidence_tier=EvidenceTier.OBSERVED,
                file_path=file_path,
                line_start=m.line_start,
                line_end=m.line_end,
                title=m.title,
                description=m.description,
                evidence=m.evidence,
                query_match_id=m.query_match_id,
                trace_path=None,
                proposal_hash=proposal_hash,
                content_hash=compute_finding_content_hash(
                    file_path=file_path,
                    line_start=m.line_start,
                    line_end=m.line_end,
                    finding_type=m.finding_type,
                ),
            )
        )
    return tuple(findings)


def compute_observed_skip_shadow(
    matches: tuple[ObservedMatch, ...],
    *,
    file_path: str,
    included_scope_units: tuple[ScopeUnit, ...],
    patched_file: PatchedFile,
    head_source: str,
    base_source: str | None,
    review_id: UUID,
    is_eval: bool,
) -> ObservedSkipShadowEvent | None:
    """Compute the per-file OBSERVED skip-eligibility decision (Cost Lever 3,
    `DECISIONS.md#049`) from the shared `run_observed_matches` records, returning
    it as an `ObservedSkipShadowEvent` — or `None` when there is nothing to
    evaluate (no included scopes, or no changed regions in them).

    Default-deny: a file is skip-eligible (`would_skip`) iff EVERY
    changed region across the included scopes lies within the coverage envelope of
    at least one `skip_safe` match. `signal_only` matches never count toward
    coverage. Base/removed regions are un-coverable by head-content matches → they
    are always blockers. Because the production registry seeds zero `skip_safe`
    queries, every production-emitted event with changed regions is `not_eligible`.
    This function only COMPUTES the decision; the caller (`analyze._process_one_file`)
    records it and, when `analyze_observed_skip_enforced` is set and the outcome is
    `would_skip`, enforces the pre-LLM skip (Step 3b-mechanism). Under the production
    defaults (flag off + zero `skip_safe`) the LLM always runs.

    The returned event is coherent by construction (it satisfies
    `ObservedSkipShadowEvent`'s coverage-coherence validator): a `would_skip` has
    no blockers, so every changed region is a covered head region → non-empty
    `covering_matches` and no base-side regions. Changed regions and blockers are
    deduped by (side, line) so overlapping included scopes don't double-count.
    """
    if not included_scope_units:
        return None

    def _region(side: Literal["head", "base"], line_no: int) -> ObservedSkipChangedRegion:
        return ObservedSkipChangedRegion(side=side, line_start=line_no, line_end=line_no)

    skip_safe_envelopes = tuple(
        (m.line_start, m.line_end) for m in matches if m.query_class == QueryClass.SKIP_SAFE
    )

    changed: list[ObservedSkipChangedRegion] = []
    blockers: list[ObservedSkipChangedRegion] = []
    seen_changed: set[tuple[str, int]] = set()
    seen_blocker: set[tuple[str, int]] = set()
    covered_head_lines: set[int] = set()

    for su in included_scope_units:
        try:
            spans = changed_line_spans(
                su, patched_file, head_source=head_source, base_source=base_source
            )
        except CoordinateError:
            # base_source=None with kept-removed lines is unreachable under the V1
            # intake contract (added files have no removed lines; modified/renamed
            # carry content_base) — the sibling trivial-scope filter pre-checks it.
            # But the shadow event is OPTIONAL telemetry, so a coordinate edge must
            # never abort the review: fail-safe to no event.
            return None
        for ls in spans.head_added:
            key = ("head", ls.line_no)
            covered = any(lo <= ls.line_no <= hi for lo, hi in skip_safe_envelopes)
            if key not in seen_changed:
                seen_changed.add(key)
                changed.append(_region("head", ls.line_no))
            if covered:
                covered_head_lines.add(ls.line_no)
            elif key not in seen_blocker:
                seen_blocker.add(key)
                blockers.append(_region("head", ls.line_no))
        for ls in spans.base_removed:
            key = ("base", ls.line_no)
            if key not in seen_changed:
                seen_changed.add(key)
                changed.append(_region("base", ls.line_no))
            # Base/removed lines are un-coverable by head-content OBSERVED matches.
            if key not in seen_blocker:
                seen_blocker.add(key)
                blockers.append(_region("base", ls.line_no))

    if not changed:
        return None

    covering = tuple(
        ObservedSkipCoveringMatch(
            query_match_id=m.query_match_id,
            side="head",
            line_start=m.line_start,
            line_end=m.line_end,
        )
        for m in matches
        if m.query_class == QueryClass.SKIP_SAFE
        and any(m.line_start <= n <= m.line_end for n in covered_head_lines)
    )

    outcome: Literal["would_skip", "not_eligible"] = (
        "would_skip" if not blockers else "not_eligible"
    )
    return ObservedSkipShadowEvent(
        review_id=review_id,
        is_eval=is_eval,
        file_path=file_path,
        outcome=outcome,
        changed_regions=tuple(changed),
        covering_matches=covering,
        blockers=tuple(blockers),
    )
