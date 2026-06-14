# Deterministic OBSERVED-tier finding producer per
# specs/2026-06-14-observed-query-library-v1.md (Cost Lever 3).
"""Turn tree-sitter security-query matches into `ReviewFinding`s with NO model
text.

A match from `queries.registry.match(observed_id, ...)` becomes a finding by a
fixed mapping: `finding_type` / `severity` / `dimension` are policy-set (never
model-set), `title` / `description` are the registry's static text, the byte
envelope maps to source lines through `coordinates.query_span_to_source_lines`,
and `evidence` is the matched source text. Construction goes through
`ReviewFinding(...)` so `enforce_proof_boundary` (OBSERVED ⇒ non-empty
`query_match_id`) and `_verify_content_hash` validate at the schema floor.

CLEAN-parse only: the caller gates on `degraded_mode` (no OBSERVED findings on a
degraded/failed parse). These are `signal_only` findings — they AUGMENT the LLM
pass and never skip it; the skip routing is a later, evidence-gated increment.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final

from outrider.audit.events import compute_finding_content_hash
from outrider.coordinates import CoordinateError, query_span_to_source_lines
from outrider.policy.canonical import compute_proposal_hash
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import lookup_severity
from outrider.queries import registry as query_registry
from outrider.schemas import ReviewFinding

if TYPE_CHECKING:
    from uuid import UUID

    from outrider.ast_facts.models import ScopeUnit


# Version of the OBSERVED producer's ADMISSION logic — the rules below that
# decide which matches become findings: the scope-CONTAINMENT predicate,
# test-file suppression (`_is_test_file`), the zero-width-span skip, and the
# byte->line mapping. Folded into the analyze cache key
# (`cache.key.compute_analyze_cache_key`) so a change to these rules invalidates
# cached analyze outcomes written under the old rules — the same staleness guard
# `ANALYZE_PARSER_VERSION` gives the LLM parser. The query `.scm` bodies + their
# registry metadata are pinned SEPARATELY by `QUERY_REGISTRY_DIGEST`; this
# constant covers only the producer's admission logic. BUMP on any rule change.
OBSERVED_PRODUCER_VERSION: Final[str] = "observed-producer-v1"


def _is_test_file(file_path: str) -> bool:
    """A repo-relative path is test code if any directory component is `tests`/
    `test`, or the filename is `conftest.py` / `test_*.py` / `*_test.py`.

    Per `docs/spec.md` §11.2: a security pattern in test code (e.g. `eval()` in a
    test fixture) is NOT a production finding. The deterministic producer can't
    make the test-context judgment the LLM does, so it suppresses OBSERVED
    findings in test files structurally.
    """
    path = PurePosixPath(file_path)
    if any(part in ("tests", "test") for part in path.parts[:-1]):
        return True
    name = path.name
    return name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py")


def produce_observed_findings(
    *,
    file_path: str,
    head_content: str,
    included_scope_units: tuple[ScopeUnit, ...],
    review_id: UUID,
    installation_id: int,
    active_policy_version: str,
) -> tuple[ReviewFinding, ...]:
    """Build deterministic OBSERVED findings for every registered OBSERVED
    query whose match falls inside an INCLUDED scope unit.

    A match is admitted only when it is CONTAINED in an included scope unit (the
    same scope discipline the LLM-citation admission uses) — a deterministic
    OBSERVED finding must anchor fully inside code the review is examining, never
    straddling a scope boundary or landing in unchanged code outside the diff.
    Test files produce nothing (spec §11.2). Returns a deterministically ordered
    tuple (by query id, then the registry's match sort) so the round's
    content-derived id stays stable across replays.
    """
    # Test code is not a production security finding (spec §11.2).
    if _is_test_file(file_path):
        return ()
    content_bytes = head_content.encode("utf-8")
    scope_ranges = tuple((su.byte_start, su.byte_end) for su in included_scope_units)
    findings: list[ReviewFinding] = []

    for query_id, observed in query_registry.OBSERVED_QUERIES.items():
        for span in query_registry.match(query_id, content_bytes):
            # A degenerate (zero-width) envelope has no reviewable line range;
            # skip rather than feed query_span_to_source_lines a span it rejects.
            if span.byte_end <= span.byte_start:
                continue
            # Containment: the match must fall fully inside an included scope's
            # byte range (a call always nests within its enclosing scope).
            if not any(s <= span.byte_start and span.byte_end <= e for s, e in scope_ranges):
                continue
            try:
                line_start, line_end = query_span_to_source_lines(
                    byte_start=span.byte_start,
                    byte_end=span.byte_end,
                    head_content=head_content,
                )
            except CoordinateError:
                # Out-of-bounds / unlocatable span — cannot anchor a finding.
                continue

            evidence = content_bytes[span.byte_start : span.byte_end].decode(
                "utf-8", errors="replace"
            )
            finding_type = observed.finding_type
            proposal_hash = compute_proposal_hash(
                source_file_path=file_path,
                finding_type=finding_type.value,
                evidence_tier=EvidenceTier.OBSERVED.value,
                query_match_id=observed.query_match_id,
                trace_path=None,
                title=observed.title,
                description=observed.description,
                evidence=evidence,
                line_start=line_start,
                line_end=line_end,
            )
            findings.append(
                ReviewFinding(
                    review_id=review_id,
                    installation_id=installation_id,
                    policy_version=active_policy_version,
                    finding_type=finding_type,
                    dimension=lookup_dimension(finding_type),
                    severity=lookup_severity(finding_type),
                    evidence_tier=EvidenceTier.OBSERVED,
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    title=observed.title,
                    description=observed.description,
                    evidence=evidence,
                    query_match_id=observed.query_match_id,
                    trace_path=None,
                    proposal_hash=proposal_hash,
                    content_hash=compute_finding_content_hash(
                        file_path=file_path,
                        line_start=line_start,
                        line_end=line_end,
                        finding_type=finding_type,
                    ),
                )
            )
    return tuple(findings)
