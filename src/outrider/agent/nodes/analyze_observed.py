# Deterministic OBSERVED-tier finding producer per
# specs/2026-06-14-observed-query-library-v1.md (Cost Lever 3).
"""Find structural security evidence without model text.

`run_observed_matches` owns query execution and ordered proof admission.
`produce_observed_findings` maps admitted matches to policy-owned findings;
`compute_observed_skip_shadow` measures coverage using only skip-safe matches.

Callers suppress queries on error-caused degraded routes. Module-level proof
requires a fully error-free parse, even when ordinary scope-contained proof
can safely ignore errors outside the included scopes. Production defaults
augment the model pass: skip enforcement is off and no query is skip-safe.
"""

from __future__ import annotations

import hashlib
import logging
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
    added_line_byte_ranges,
    changed_line_spans,
    query_span_to_source_lines,
)
from outrider.policy.canonical import compute_proposal_hash
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import lookup_severity
from outrider.queries import registry as query_registry
from outrider.queries.observed import (
    ANCHOR_CAPTURE_PREFERENCE,
    GUARD_POSITION_CAPTURES,
    QueryClass,
)
from outrider.schemas import ReviewFinding

if TYPE_CHECKING:
    from uuid import UUID

    from unidiff import PatchedFile

    from outrider.ast_facts.models import (
        ImportRef,
        LexicalBinding,
        ParseResult,
        QueryCaptureSpan,
        QueryMatchSpan,
        ScopeUnit,
    )
    from outrider.policy.severity import FindingType
    from outrider.queries.observed import BindingRule, ObservedQuery, QueryLanguage


# Controls cache invalidation when `run_observed_matches` changes which query
# matches are admitted (`cache.key.compute_analyze_cache_key`).
# Bump on any admission-rule change: language selection, test-file suppression,
# scope containment (including module scope), import/value-binding checks,
# lexical shadowing, zero-width handling, or byte-to-line mapping.
# Binding/shadowing: DECISIONS.md#060; module scope: DECISIONS.md#062.
# Query bodies and registry metadata use QUERY_REGISTRY_DIGEST separately.
OBSERVED_PRODUCER_VERSION: Final[str] = "observed-producer-v7"

# Cap source evidence at `ReviewFinding.evidence`'s schema limit so long query
# matches still construct a valid finding.
_EVIDENCE_MAX_CHARS: Final[int] = 2000

logger = logging.getLogger(__name__)


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
    # True when the match was admitted by the MODULE-SCOPE arm (scope-disjoint,
    # diff-anchored — DECISIONS.md#062) rather than scope containment. Consumed
    # by `compute_observed_skip_shadow` to exclude module-arm matches from #049
    # skip coverage: their proof is diff-anchored, not scope-coverage-shaped.
    module_level: bool = False


def _is_test_file(file_path: str, language: QueryLanguage | None) -> bool:
    """Suppress test fixtures as production findings (spec §11.2).

    `test`/`tests` directories and Python test filenames apply to all paths.
    JS/TS additionally recognizes `__tests__` and inner `.test`/`.spec`
    segments; `report.spec.py` remains production Python code.
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
    return any(segment in ("test", "spec") for segment in inner_segments)


def _module_matches(specifier: str, modules: tuple[str, ...]) -> bool:
    """Match a package or its slash-delimited subpath, never a lookalike.

    `mysql2/promise` matches `mysql2`; `mysql2-mock` does not.
    """
    return any(specifier == module or specifier.startswith(module + "/") for module in modules)


def _module_presence_satisfied(
    modules: tuple[str, ...], import_refs: tuple[ImportRef, ...]
) -> bool:
    """Share value-import admission between query prefilter and span checks."""
    return any(ref.is_value_import and _module_matches(ref.module, modules) for ref in import_refs)


def _shadowed(
    name: str,
    match_byte_start: int,
    match_byte_end: int,
    lexical_bindings: tuple[LexicalBinding, ...],
) -> bool:
    """A local binding shadows the name only while its span contains the match.

    A catch binding cannot shadow a later call or a sibling function.
    """
    return any(
        binding.name == name
        and binding.visibility_byte_start <= match_byte_start
        and match_byte_end <= binding.visibility_byte_end
        for binding in lexical_bindings
    )


def _guarded_global_shadowed(
    shadow_guard: tuple[str, ...],
    captures: tuple[QueryCaptureSpan, ...],
    content_bytes: bytes,
    match_byte_start: int,
    match_byte_end: int,
    lexical_bindings: tuple[LexicalBinding, ...],
    import_refs: tuple[ImportRef, ...],
) -> bool:
    """Reject a guarded callee/receiver rebound by a local or value import.

    Argument captures do not participate: a shadowed `Function` argument in
    `eval(Function)` cannot shadow the `eval` callee. Type imports and
    re-exports create no runtime binding. The import check is file-level;
    function-local require bindings can over-deny sibling uses (FUP-214).
    """
    if not shadow_guard:
        return False
    capture_texts = {
        content_bytes[capture.byte_start : capture.byte_end].decode("utf-8", errors="replace")
        for capture in captures
        if capture.name in GUARD_POSITION_CAPTURES
    }
    imported_names = {name for ref in import_refs if ref.is_value_import for name in ref.names}
    return any(
        guarded_name in capture_texts
        and (
            _shadowed(guarded_name, match_byte_start, match_byte_end, lexical_bindings)
            or guarded_name in imported_names
        )
        for guarded_name in shadow_guard
    )


def _binding_admits(
    binding: BindingRule | None,
    captures: tuple[QueryCaptureSpan, ...],
    content_bytes: bytes,
    import_refs: tuple[ImportRef, ...],
    *,
    lexical_bindings: tuple[LexicalBinding, ...],
    match_byte_start: int,
    match_byte_end: int,
) -> bool:
    """Prove that a name-anchored match resolves to the rule's dangerous API.

    `anchor_import` requires an unshadowed value-import binding, preferring
    receiver over callee. Missing captures deny admission. `module_presence`
    requires a value import anywhere in the file; this can over-admit an
    unrelated receiver (FUP-214). No binding rule admits on structure alone,
    with guarded globals checked separately.
    """
    if binding is None:
        return True
    if binding.mode == "module_presence":
        return _module_presence_satisfied(binding.modules, import_refs)
    if binding.mode == "anchor_import":
        # A receiver identifies the API before a callee name does.
        anchor_span = next(
            (
                capture
                for name in ANCHOR_CAPTURE_PREFERENCE
                for capture in captures
                if capture.name == name
            ),
            None,
        )
        if anchor_span is None:
            return False
        anchor = content_bytes[anchor_span.byte_start : anchor_span.byte_end].decode(
            "utf-8", errors="replace"
        )
        if _shadowed(anchor, match_byte_start, match_byte_end, lexical_bindings):
            return False
        return any(
            ref.is_value_import
            and _module_matches(ref.module, binding.modules)
            and anchor in ref.names
            for ref in import_refs
        )
    assert_never(binding.mode)


def import_bindings_digest(import_refs: tuple[ImportRef, ...]) -> str:
    """Pin imports that influence admission but may be absent from the prompt.

    Hash the order/duplicate-insensitive set of module, value marker and names.
    Count headers and length prefixes preserve boundaries; kind, line and path
    are ignored. The Python-shaped `from_import_map_digest` misses JS/TS inputs.

    Exact names and package subpaths deliberately over-invalidate compared with
    module-presence admission. This costs hits without risking stale findings;
    `QUERY_REGISTRY_DIGEST` separately pins the rules.
    """
    entries = sorted(
        {(ref.module, ref.is_value_import, tuple(sorted(ref.names))) for ref in import_refs}
    )
    hasher = hashlib.sha256()
    hasher.update(f"{len(entries)}:".encode())
    for module, is_value_import, names in entries:
        module_bytes = module.encode("utf-8")
        hasher.update(f"{len(module_bytes)}:".encode())
        hasher.update(module_bytes)
        hasher.update(b"1:" if is_value_import else b"0:")
        hasher.update(f"{len(names)}:".encode())
        for name in names:
            name_bytes = name.encode("utf-8")
            hasher.update(f"{len(name_bytes)}:".encode())
            hasher.update(name_bytes)
    return hasher.hexdigest()


def lexical_bindings_digest(lexical_bindings: tuple[LexicalBinding, ...]) -> str:
    """Pin shadowing bindings that may live outside the rendered scopes.

    Hash unique names and visibility spans, ignoring kind, line and path.
    Earlier byte edits can shift spans and cost cache hits without staleness.
    """
    entries = sorted(
        {
            (binding.name, binding.visibility_byte_start, binding.visibility_byte_end)
            for binding in lexical_bindings
        }
    )
    hasher = hashlib.sha256()
    hasher.update(f"{len(entries)}:".encode())
    for name, start, end in entries:
        name_bytes = name.encode("utf-8")
        hasher.update(f"{len(name_bytes)}:".encode())
        hasher.update(name_bytes)
        hasher.update(f"{start}:{end}:".encode())
    return hasher.hexdigest()


def module_admission_digest(
    added_line_ranges: tuple[tuple[int, int], ...],
    all_scope_units: tuple[ScopeUnit, ...],
    content_bytes: bytes,
) -> str:
    """Pin changed ranges, every parsed scope span, and changed module bytes.

    Ranges retain patch order; scope spans are sorted and deduplicated.
    Content matters because identical ranges over different bytes produce
    different query matches (DECISIONS.md#062).
    """
    hasher = hashlib.sha256()
    hasher.update(f"{len(added_line_ranges)}:".encode())
    for range_start, range_end in added_line_ranges:
        covered = content_bytes[range_start:range_end]
        hasher.update(f"{range_start}:{range_end}:{len(covered)}:".encode())
        hasher.update(covered)
    spans = sorted({(scope.byte_start, scope.byte_end) for scope in all_scope_units})
    hasher.update(f"{len(spans)}:".encode())
    for span_start, span_end in spans:
        hasher.update(f"{span_start}:{span_end}:".encode())
    return hasher.hexdigest()


def _has_clean_module_parse(parse_result: ParseResult) -> bool:
    """Module proof needs reliable scope boundaries across the entire file."""
    if parse_result.parser_outcome != "clean":
        return False
    if parse_result.error_lines:
        return False
    return not any(parse_result.has_error.values())


def module_admission_inputs(
    parse_result: ParseResult,
    patched_file: PatchedFile | None,
    head_content: str | None,
) -> tuple[tuple[ScopeUnit, ...], tuple[tuple[int, int], ...]]:
    """Derive changed-file proof inputs for module-level matches (#062).

    A fully error-free parse makes scope disjointness trustworthy. Missing
    patch/content or misaligned coordinates deny admission; misalignment is
    logged without aborting. The node normally skips it earlier (FUP-217).
    Production callers use this gate or its whole-file sibling rather than
    passing unchecked proof inputs directly to `run_observed_matches`.
    """
    if patched_file is None or head_content is None or not _has_clean_module_parse(parse_result):
        return (), ()
    try:
        ranges = added_line_byte_ranges(patched_file, head_content)
    except CoordinateError:
        logger.warning(
            "module-scope arm: patch/head-content misalignment for %s — "
            "module admission inert for this file (review continues)",
            patched_file.path,
        )
        return (), ()
    if not ranges:
        return (), ()
    return tuple(parse_result.scope_units), ranges


def module_admission_inputs_whole_file(
    parse_result: ParseResult, content_bytes: bytes
) -> tuple[tuple[ScopeUnit, ...], tuple[tuple[int, int], ...]]:
    """Apply the same proof gate with the whole file treated as changed."""
    if not _has_clean_module_parse(parse_result) or not content_bytes:
        return (), ()
    return tuple(parse_result.scope_units), ((0, len(content_bytes)),)


def _module_level_admits(
    observed: ObservedQuery,
    span: QueryMatchSpan,
    *,
    all_scope_ranges: tuple[tuple[int, int], ...],
    added_line_ranges: tuple[tuple[int, int], ...],
) -> bool:
    """Admit an eligible query only outside every parsed scope and inside a diff.

    Absence from included scopes alone proves nothing: unchanged scopes also
    exclude module-level proof. Empty changed ranges deny admission (#062).
    The caller still checks shadowing, binding and source coordinates.
    """
    if not observed.module_scope_eligible or not added_line_ranges:
        return False
    for scope_start, scope_end in all_scope_ranges:
        if not (span.byte_end <= scope_start or scope_end <= span.byte_start):
            return False
    return any(
        added_start <= span.byte_start and span.byte_end <= added_end
        for added_start, added_end in added_line_ranges
    )


def run_observed_matches(
    *,
    file_path: str,
    head_content: str,
    included_scope_units: tuple[ScopeUnit, ...],
    import_refs: tuple[ImportRef, ...],
    lexical_bindings: tuple[LexicalBinding, ...],
    all_scope_units: tuple[ScopeUnit, ...] = (),
    added_line_ranges: tuple[tuple[int, int], ...] = (),
) -> tuple[ObservedMatch, ...]:
    """Run the file's registered queries and return admitted structural matches.

    Test files and unregistered languages return nothing. Each match needs a
    positive span, proof within an included scope or the module-level diff
    gate, valid global/import bindings, and mappable source coordinates.
    Module proof inputs must come through `module_admission_inputs` or its
    whole-file sibling; empty inputs deny that route.

    Preserve catalog order then registry match order for stable replay IDs.
    """
    language = query_registry.query_language_for_path(file_path)
    grammar = query_registry.grammar_for_path(file_path)
    if language is None or grammar is None:
        return ()
    if _is_test_file(file_path, language):
        return ()
    content_bytes = head_content.encode("utf-8")
    scope_ranges = tuple((scope.byte_start, scope.byte_end) for scope in included_scope_units)
    all_scope_ranges = tuple((scope.byte_start, scope.byte_end) for scope in all_scope_units)
    matches: list[ObservedMatch] = []

    for query_id, observed in query_registry.observed_queries_for(language).items():
        # Missing imports can rule out a query before tree-sitter runs.
        # Reuse the span-admission predicate to keep this prefilter consistent.
        if (
            observed.binding is not None
            and observed.binding.mode == "module_presence"
            and not _module_presence_satisfied(observed.binding.modules, import_refs)
        ):
            continue
        for span in query_registry.match(query_id, content_bytes, grammar=grammar):
            if span.byte_end <= span.byte_start:
                continue
            # Proof needs an included scope or a disjoint, diff-anchored span.
            contained = any(
                scope_start <= span.byte_start and span.byte_end <= scope_end
                for scope_start, scope_end in scope_ranges
            )
            if not contained and not _module_level_admits(
                observed,
                span,
                all_scope_ranges=all_scope_ranges,
                added_line_ranges=added_line_ranges,
            ):
                continue
            # Captured globals and imported APIs must resolve to their real bindings.
            if _guarded_global_shadowed(
                observed.shadow_guard,
                span.captures,
                content_bytes,
                span.byte_start,
                span.byte_end,
                lexical_bindings,
                import_refs,
            ):
                continue
            if not _binding_admits(
                observed.binding,
                span.captures,
                content_bytes,
                import_refs,
                lexical_bindings=lexical_bindings,
                match_byte_start=span.byte_start,
                match_byte_end=span.byte_end,
            ):
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
                    module_level=not contained,
                )
            )
    return tuple(matches)


def module_level_observed_matches(
    *,
    file_path: str,
    head_content: str,
    all_scope_units: tuple[ScopeUnit, ...],
    added_line_ranges: tuple[tuple[int, int], ...],
    import_refs: tuple[ImportRef, ...],
    lexical_bindings: tuple[LexicalBinding, ...],
) -> tuple[ObservedMatch, ...]:
    """Run only module-level proof, reusing the ordinary admission chain (#062).

    Empty included scopes leave only diff-anchored, scope-disjoint matches.
    Languages without an eligible query avoid the sweep. Callers can retain
    these findings when scope routing, budget or triviality would skip a file.
    """
    language = query_registry.query_language_for_path(file_path)
    if language is None or not any(
        query.module_scope_eligible
        for query in query_registry.observed_queries_for(language).values()
    ):
        return ()
    return run_observed_matches(
        file_path=file_path,
        head_content=head_content,
        included_scope_units=(),
        import_refs=import_refs,
        lexical_bindings=lexical_bindings,
        all_scope_units=all_scope_units,
        added_line_ranges=added_line_ranges,
    )


def produce_observed_findings(
    matches: tuple[ObservedMatch, ...],
    *,
    file_path: str,
    review_id: UUID,
    installation_id: int,
    active_policy_version: str,
) -> tuple[ReviewFinding, ...]:
    """Map every admitted match to a policy-owned finding in the same order.

    Both signal-only and skip-safe matches emit; query class affects routing.
    Schema construction validates proof and content hashes.
    """
    findings: list[ReviewFinding] = []
    for match in matches:
        proposal_hash = compute_proposal_hash(
            source_file_path=file_path,
            finding_type=match.finding_type.value,
            evidence_tier=EvidenceTier.OBSERVED.value,
            query_match_id=match.query_match_id,
            trace_path=None,
            title=match.title,
            description=match.description,
            evidence=match.evidence,
            line_start=match.line_start,
            line_end=match.line_end,
        )
        findings.append(
            ReviewFinding(
                review_id=review_id,
                installation_id=installation_id,
                policy_version=active_policy_version,
                finding_type=match.finding_type,
                dimension=lookup_dimension(match.finding_type),
                severity=lookup_severity(match.finding_type),
                evidence_tier=EvidenceTier.OBSERVED,
                file_path=file_path,
                line_start=match.line_start,
                line_end=match.line_end,
                title=match.title,
                description=match.description,
                evidence=match.evidence,
                query_match_id=match.query_match_id,
                trace_path=None,
                proposal_hash=proposal_hash,
                content_hash=compute_finding_content_hash(
                    file_path=file_path,
                    line_start=match.line_start,
                    line_end=match.line_end,
                    finding_type=match.finding_type,
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
    """Measure skip eligibility without recording or enforcing the decision.

    Every changed line in the included scopes needs skip-safe head coverage;
    removed lines always block. Module-level matches cannot cover scope edits.
    Overlapping scopes are deduplicated by side and line. Empty scopes/changes
    or unusable coordinates produce no optional event.

    The caller records the result and may enforce `would_skip`. Production
    defaults disable enforcement and register no skip-safe queries (#049).
    """
    if not included_scope_units:
        return None

    def _region(side: Literal["head", "base"], line_no: int) -> ObservedSkipChangedRegion:
        return ObservedSkipChangedRegion(side=side, line_start=line_no, line_end=line_no)

    # Disjoint byte spans can share a source line. Module-level proof must not
    # accidentally count as coverage of an included scope (#049, #062).
    skip_safe_envelopes = tuple(
        (match.line_start, match.line_end)
        for match in matches
        if match.query_class == QueryClass.SKIP_SAFE and not match.module_level
    )

    changed: list[ObservedSkipChangedRegion] = []
    blockers: list[ObservedSkipChangedRegion] = []
    seen_changed: set[tuple[str, int]] = set()
    seen_blocker: set[tuple[str, int]] = set()
    covered_head_lines: set[int] = set()

    for scope in included_scope_units:
        try:
            spans = changed_line_spans(
                scope, patched_file, head_source=head_source, base_source=base_source
            )
        except CoordinateError:
            # Optional coverage telemetry must not abort a review.
            return None
        for line_span in spans.head_added:
            key = ("head", line_span.line_no)
            covered = any(lo <= line_span.line_no <= hi for lo, hi in skip_safe_envelopes)
            if key not in seen_changed:
                seen_changed.add(key)
                changed.append(_region("head", line_span.line_no))
            if covered:
                covered_head_lines.add(line_span.line_no)
            elif key not in seen_blocker:
                seen_blocker.add(key)
                blockers.append(_region("head", line_span.line_no))
        for line_span in spans.base_removed:
            key = ("base", line_span.line_no)
            if key not in seen_changed:
                seen_changed.add(key)
                changed.append(_region("base", line_span.line_no))
            # Base/removed lines are un-coverable by head-content OBSERVED matches.
            if key not in seen_blocker:
                seen_blocker.add(key)
                blockers.append(_region("base", line_span.line_no))

    if not changed:
        return None

    covering = tuple(
        ObservedSkipCoveringMatch(
            query_match_id=match.query_match_id,
            side="head",
            line_start=match.line_start,
            line_end=match.line_end,
        )
        for match in matches
        if match.query_class == QueryClass.SKIP_SAFE
        and any(match.line_start <= line_no <= match.line_end for line_no in covered_head_lines)
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
