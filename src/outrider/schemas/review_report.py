# Synthesize-output cross-boundary models per docs/spec.md §7.2 (lines 1099-1114).
"""Synthesize envelope: ReviewMetrics / ReviewReport.

These models are the typed output contract of the synthesize node (specs/
2026-05-28-synthesize-node.md). Synthesize aggregates findings from all
analysis rounds into a single ReviewReport, computes deterministic metrics
from audit_events + analysis_rounds, and runs one Sonnet call for the
free-form summary prose. Downstream nodes (hitl, publish) consume the
ReviewReport; they do NOT walk state.analysis_rounds[*].findings any more.

ReviewReport.summary is Sonnet output: untrusted prose rendered into the
GitHub review body downstream. Per docs/trust-boundaries.md §6, the
deterministic publish-time gate (policy/output_sanitizer.py::
sanitize_display_string + apply_size_cap) sanitizes the prose at the
review-body builder. Field-level Field(max_length=2000) is the schema-side
codepoint cap that mirrors spec.md:1101 — note that this counts codepoints,
not graphemes or UTF-8 bytes; publish-time `apply_size_cap` is the
authoritative byte-budget gate. Sanitization happens at publish, not at
synthesize-node construction.

ReviewReport.findings is the deduplicated set keyed by content_hash. Same-
content_hash + same finding_type + same policy_version => same severity by
construction (severity-set-by-policy + severity-policy-versioned-for-replay
+ ReviewFinding._verify_baseline_severity + compute_finding_content_hash
recipe). Synthesize fails loud on cross-round divergence for the same
content_hash on EITHER axis (severity OR policy_version — both are
corruption signals, same recovery action) via SynthesizeAggregationError +
paired AnomalySink.emit_anomaly(rule_name=CROSS_ROUND_SEVERITY_DIVERGENCE)
emission. See pre-spec gate #7 in specs/_2026-05-27-synthesize-pre-spec-
gates.md.

Order of findings is canonicalized at the schema layer per spec.md:1103
("deduplicated, sorted by severity"). The `_canonicalize_findings`
validator sorts by `(severity_sort_key, file_path, line_start, line_end,
finding_id)` — severity descending (CRITICAL first), then by location for
within-tier stability. Producer order is irrelevant; the schema produces
the canonical order. Sibling precedent:
`TriageResult._canonicalize_relevant_dimensions` (canonical-sorts
relevant_dimensions for the same JSON-payload-identity reason). Two
ReviewReport instances built from the same set of findings (regardless of
emission order from analyze rounds) serialize to byte-identical JSON.

Canonical-shape note: spec.md:1103 declares `findings: list[ReviewFinding]`,
but project convention favors `tuple[ReviewFinding, ...]` for true
immutability (matches AnalysisRound.findings, HITLRequest fields, PRContext
.changed_files). Spec amendment route taken: deviate from canonical list →
tuple here, flagged in pre-spec gate-resolution doc for upstream amendment
when canonical record is next revised. Other call-sites already drift the
same way (AnalysisRound, HITLRequest) — this is uniform project-side
convention, not a one-off divergence.

policy_version is intentionally OMITTED from ReviewReport per pre-spec gate
#1: the field is scoped to SynthesizeCompletedEvent (audit-event mirror)
and per-finding ReviewFinding.policy_version, not promoted to the
ReviewReport state slot. This is the lower-risk amendment route — the
canonical-shape change to spec.md is deferred until a future spec arc
revisits it. Per-finding versioning + audit-event versioning together
provide the replay-correctness the `severity-policy-versioned-for-replay`
invariant requires.

Frozen=True on both models: ReviewReport rides on ReviewState through every
LangGraph checkpoint after synthesize lands; mutation after construction
would break checkpoint-replay equivalence. NOTE: frozen on the envelope is
SHALLOW — ReviewFinding is intentionally NOT frozen (uses
validate_assignment=True per its own module docstring). A holder of
`report.findings[i]` can still execute `report.findings[i].severity = ...`
and the assignment runs through ReviewFinding's validator chain. Treat
findings inside a ReviewReport as logically-immutable-by-convention; do
NOT mutate them downstream. If a downstream node needs a modified version
of a finding (e.g., publish_destination set), construct a fresh
ReviewFinding via `model_validate({**finding.model_dump(), **{...}})`
rather than mutating in place.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from outrider.policy.severity import FindingSeverity
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.triage_result import RiskLevel

# Severity presentation order: CRITICAL first (most-severe at the top of
# review-body lists, HITL-partition gated-set, dashboard). Wrapped in
# MappingProxyType so runtime mutation raises TypeError — same defense-
# in-depth shape as policy/severity.py::SEVERITY_POLICY. Inlined here
# rather than in policy/severity.py because this is a presentation-layer
# concern (sort order for review output), not a severity-from-policy
# concern. Module-private: importers should use the public sorted tuple
# on ReviewReport, not the order map directly.
_SEVERITY_SORT_KEY: Final[Mapping[FindingSeverity, int]] = MappingProxyType(
    {
        FindingSeverity.CRITICAL: 0,
        FindingSeverity.HIGH: 1,
        FindingSeverity.MEDIUM: 2,
        FindingSeverity.LOW: 3,
        FindingSeverity.INFO: 4,
    }
)


class ReviewMetrics(BaseModel):
    """Per-review statistics computed in synthesize, per spec.md:1106-1114.

    All fields derived deterministically from audit_events + analysis_rounds
    + node wall-clock measurement. NOT computed by the Sonnet call; the
    Sonnet call produces only the summary prose. ge=0 floors match the
    LLMCallEvent pricing-field convention (input_tokens, output_tokens,
    cost_usd) at audit/events.py:320-323.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Deterministically computed from state.analysis_rounds.
    files_examined: int = Field(ge=0)
    # Deterministically computed from
    # `state.trace_decisions[*].(target_file | resolved_candidate_paths)`
    # ∪ `state.trace_fetched_files[*].path`, minus `pr_context.changed_files`
    # paths. See `_compute_files_traced_beyond_diff` for the union recipe
    # and the "beyond diff = outside changed-files set, NOT
    # Phase-2-fetched specifically" semantic.
    files_traced_beyond_diff: int = Field(ge=0)
    # LLM-aggregate metrics. V1 placeholder semantics:
    # `None` indicates "audit-query helper not yet wired" rather than
    # "zero." The audit truth for these aggregates lives in
    # `audit_events`: sum `LLMCallEvent.input_tokens` /
    # `LLMCallEvent.output_tokens` / `LLMCallEvent.cost_usd` joined on
    # `review_id`. The dashboard reads audit truth; ReviewMetrics
    # is a denormalized convenience snapshot. FUP: wire the audit-query
    # helper + populate these from the helper at synthesize-emit time.
    # Optional+None makes the "unknown vs. zero" distinction explicit
    # on the audit row — a placeholder-zero would land as durable
    # false metadata.
    llm_calls_made: int | None = Field(default=None, ge=0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    # Upper cap defends against `float('inf')` propagating into Postgres
    # JSONB (some JSONB configs reject non-standard JSON `Infinity`).
    # le=100.0 is "this would already be a runaway"; real V1 reviews land
    # well under $1. Cap is policy-driven, not architectural — bump if
    # average cost rises. Optional+None per the same V1-placeholder
    # rationale as the token fields above.
    total_cost_usd: float | None = Field(default=None, ge=0, le=100.0)
    # Wall-clock IS deterministically computable from node-side
    # time.monotonic() delta — not a placeholder.
    # le=86400 (24h) bounds wall-clock to a single day; HITL-paused
    # reviews use `state.received_at` + HITL expiry rather than letting
    # the synthesize wall-clock balloon. A multi-day review is a bug,
    # not a workload.
    wall_clock_seconds: float = Field(ge=0, le=86400)


class ReviewReport(BaseModel):
    """Output of the synthesize node, per spec.md:1099-1104.

    `findings` is the deduplicated set across all analysis rounds, keyed by
    content_hash. The canonical-sort validator enforces deterministic
    JSON-payload identity for checkpoint comparison + audit content-hashing:
    same logical findings serialize to the same bytes regardless of which
    round emitted each instance.

    Duplicate content_hash entries are rejected at the schema layer (defense
    in depth on top of synthesize's node-side dedup). A producer that
    constructs a ReviewReport with two findings sharing a content_hash is
    bypassing synthesize's dedup contract — fail-loud here surfaces the bug
    rather than silently accepting it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str = Field(max_length=2000)
    overall_risk: RiskLevel
    # max_length cap follows AnalysisRound.findings precedent (50 per
    # round) with cross-round aggregate headroom. V1 caps analyze at
    # MAX_ANALYSIS_ROUNDS=2 → 100 raw findings maximum, post-dedup
    # typically much fewer. The 200 cap is "this would already be a
    # runaway" defense — protects checkpoint payload size + downstream
    # HITL-partition pagination + audit-row JSONB payload. Bump if real
    # workloads exceed it; the bump itself is a one-line policy edit.
    findings: tuple[ReviewFinding, ...] = Field(max_length=200)
    metrics: ReviewMetrics

    @field_validator("findings", mode="after")
    @classmethod
    def _canonicalize_findings(cls, value: tuple[ReviewFinding, ...]) -> tuple[ReviewFinding, ...]:
        """Reject duplicate content_hashes AND return a canonically sorted
        tuple — synthesize's dedup + spec.md:1103 "sorted by severity"
        contract.

        Sibling-precedent: TriageResult._canonicalize_relevant_dimensions
        rejects duplicates with the same fail-loud rationale (silent dedup
        masks producer bugs) AND returns sorted tuple for JSON-payload
        identity. A ReviewReport with duplicate content_hashes means
        synthesize's node-side dedup did not run, OR a direct constructor
        bypassed it (replay path, test fixture, future producer bug).

        Sort key: (severity_sort_key, file_path, line_start, line_end,
        finding_id) — CRITICAL first then HIGH/MEDIUM/LOW/INFO; within a
        severity tier, sort by location for stable presentation; finding_id
        last as a final deterministic tie-breaker on the per-emission UUID.
        """
        hashes = [finding.content_hash for finding in value]
        if len(hashes) != len(set(hashes)):
            duplicates = sorted({h for h in hashes if hashes.count(h) > 1})
            raise ValueError(
                f"ReviewReport.findings contains duplicate content_hashes: "
                f"{duplicates!r}; synthesize.dedup_findings should have "
                f"collapsed these to one representative. Direct construction "
                f"bypassing synthesize's dedup is a producer bug — fail-loud "
                f"rather than silently accepting duplicates that break the "
                f"dedup invariant the rest of the pipeline depends on."
            )
        return tuple(
            sorted(
                value,
                key=lambda f: (
                    _SEVERITY_SORT_KEY[f.severity],
                    f.file_path,
                    f.line_start,
                    f.line_end,
                    str(f.finding_id),
                ),
            )
        )


__all__ = [
    "ReviewMetrics",
    "ReviewReport",
]
