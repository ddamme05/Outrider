# Synthesize-output cross-boundary models per docs/spec.md §7.2 (lines 1099-1114).
"""Synthesize's sorted findings, summary prose, and measured review metrics.

HITL and publish consume this report rather than walking analysis rounds.
Synthesize deduplicates by content_hash and rejects cross-round severity or
policy-version divergence. The schema rejects duplicate hashes and sorts
findings by severity, location, and finding_id for stable checkpoint bytes.

Summary is untrusted LLM prose, displayed on the dashboard; publish currently
builds its GitHub review body from deterministic finding sections. Metrics
come from state and audit aggregates, not numbers in summary text. Summary
model routing follows DECISIONS.md#043; retained audit metrics remain separate
from expiring content per DECISIONS.md#014.

The findings tuple is permanent per
`DECISIONS.md#030-reviewreport-tuple-not-list-findings-field`. The frozen
envelope is shallow: nested ReviewFinding objects remain lifecycle-mutable.
Treat report findings as logically immutable; clone through model_validate
for downstream changes. Per-finding policy_version and the synthesize audit
event carry the snapshot for severity-policy-versioned-for-replay; the
report deliberately has no policy_version field. Baseline severity remains
severity-set-by-policy."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from outrider.policy.severity import FindingSeverity
from outrider.schemas.analysis_round import MAX_FINDINGS_HARD_CAP
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

# Per-report finding ceilings (FUP-180), mirroring `AnalysisRound`'s two-tier shape.
# `MAX_FINDINGS_PER_REPORT` is the SOFT cap: synthesize truncates NON-gated findings
# across all rounds to this bound before building the report (via
# `agent.nodes.finding_cap`). Gated (CRITICAL/HIGH) findings are NEVER dropped to fit
# it, so the report can legitimately exceed it; the schema `max_length` below is the
# shared HARD runaway ceiling (`MAX_FINDINGS_HARD_CAP`). Since the cap runs before
# `ReviewReport(...)`, the schema constraint is the defense-in-depth backstop.
MAX_FINDINGS_PER_REPORT: Final[int] = 200


class ReviewMetrics(BaseModel):
    """Per-review statistics computed in synthesize, per spec.md:1106-1114.

    All fields derived deterministically from audit_events + analysis_rounds
    + node wall-clock measurement. NOT computed by the LLM call; the
    summary call produces only the summary prose. ge=0 floors match the
    LLMCallEvent pricing-field convention (input_tokens, output_tokens,
    cost_usd) at audit/events.py:320-323.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # See DECISIONS.md#030-reviewreport-tuple-not-list-findings-field
    # for the canonical-record amendment establishing:
    #   (a) `ReviewReport.findings: tuple[...]` as the permanent shape
    #       (vs spec.md §7.3's `list[...]` — unmaintained drift), AND
    #   (b) `ReviewMetrics` LLM-aggregate fields as `int | None` /
    #       `float | None` — populated from the audit stream (FUP-093) but
    #       kept nullable for append-only historical-row read-compat
    #       (#030 amended 2026-06-03; the planned type-revert was superseded).
    # The `files_traced_beyond_diff` union recipe Pass-1-folded under
    # #030 is documented at `_compute_files_traced_beyond_diff` and
    # pinned by `tests/unit/test_synthesize_files_traced_metric.py`.

    # Deterministically computed from state.analysis_rounds.
    files_examined: int = Field(ge=0)
    # Deterministically computed from
    # `state.trace_decisions[*].(target_file | resolved_candidate_paths)`
    # ∪ `state.trace_fetched_files[*].path`, minus `pr_context.changed_files`
    # paths. See `_compute_files_traced_beyond_diff` for the union recipe
    # and the "beyond diff = outside changed-files set, NOT
    # Phase-2-fetched specifically" semantic.
    files_traced_beyond_diff: int = Field(ge=0)
    # LLM-aggregate metrics. Populated at synthesize-emit time from the
    # audit-stream SUM over this review's `LLMCallEvent` rows
    # (`audit/aggregates.py::aggregate_review_llm_metrics` — the same
    # single-source helper the dashboard read-API uses, FUP-093). Kept
    # `int | None` / `float | None` NOT because they're unpopulated, but for
    # append-only read-compat: pre-FUP-093 rows serialize `null` here and
    # replay re-validates historical payloads through the strict adapter, so a
    # required type would reject them (#030, amended). A `None` now means
    # "historical row, predates population," never the false-zero #030
    # rejected. New rows always carry the real aggregate.
    llm_calls_made: int | None = Field(default=None, ge=0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    # Upper cap defends against `float('inf')` propagating into Postgres
    # JSONB (some JSONB configs reject non-standard JSON `Infinity`).
    # le=100.0 is "this would already be a runaway"; real V1 reviews land
    # well under $1. Cap is policy-driven, not architectural — bump if
    # average cost rises. Optional+None per the same read-compat
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

    # PROSE, not an audit-counting surface: numbers inside the summary are
    # model-written and may drift on DERIVED counts (the prompt supplies a
    # precomputed n_findings, but e.g. distinct-file counts are re-derived
    # by the model — both Sonnet and Haiku miscounted one in the 2026-06-10
    # side-by-side). Renderers display exact counts from `metrics` /
    # the audit stream; never parse them out of this text.
    summary: str = Field(max_length=2000)
    overall_risk: RiskLevel
    # max_length is the HARD runaway ceiling (MAX_FINDINGS_HARD_CAP), NOT the soft
    # MAX_FINDINGS_PER_REPORT — gated findings are never dropped to fit the soft cap
    # (FUP-180), so the report can legitimately exceed it. Synthesize re-caps the
    # deduped cross-round union severity-ordered BEFORE building the report, so this
    # constraint is the runaway-protection backstop (checkpoint payload size +
    # HITL-partition pagination + the audit-row JSONB payload).
    findings: tuple[ReviewFinding, ...] = Field(max_length=MAX_FINDINGS_HARD_CAP)
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
