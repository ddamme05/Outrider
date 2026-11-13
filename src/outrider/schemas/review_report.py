"""Collect the final summary, findings and metrics for approval and publication.

Findings have stable order and unique identities. The report is frozen and
holds a tuple, but its findings remain mutable. Clone and revalidate findings
before downstream changes. Summary text remains untrusted display data;
recorded policy versions preserve the meaning of saved severities.

See DECISIONS.md#014, #030, #043 and severity-policy-versioned-for-replay.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from outrider.policy.severity import FindingSeverity
from outrider.schemas.analysis_round import MAX_FINDINGS_HARD_CAP
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.triage_result import RiskLevel

# Immutable presentation order, separate from policy's severity assignment.
# Consumers use the sorted ReviewReport.findings tuple rather than this map.
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
# `MAX_FINDINGS_PER_REPORT` is the SOFT cap: synthesize truncates unprotected findings
# across all rounds to this bound before building the report (via
# `agent.nodes.finding_cap`). CRITICAL/HIGH and known held findings survive this
# limit, so the report can exceed it; the schema `max_length` below is the
# shared HARD runaway ceiling (`MAX_FINDINGS_HARD_CAP`). Since the cap runs before
# `ReviewReport(...)`, the schema constraint is the defense-in-depth backstop.
MAX_FINDINGS_PER_REPORT: Final[int] = 200

# Persisted summary sources name immutable renderers so audit can verify them.
EMPTY_FINDINGS_SUMMARY_SOURCE: Final = "empty-findings-v1"


def render_empty_findings_summary(overall_risk: RiskLevel) -> str:
    """State the retained count and upstream risk without inferring clean code."""
    return (
        "This review retained no findings. "
        f"Triage classified the pull request as {overall_risk.value} risk."
    )


class ReviewMetrics(BaseModel):
    """Per-review statistics computed in synthesize, per spec.md:1106-1114.

    All fields derived deterministically from audit_events + analysis_rounds
    + node wall-clock measurement. NOT computed by the LLM call; the
    optional summary call produces only summary prose. ge=0 floors match the
    LLMCallEvent pricing-field convention (input_tokens, output_tokens,
    cost_usd) at audit/events.py:320-323.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # DECISIONS.md#030-reviewreport-tuple-not-list-findings-field fixes the report's
    # tuple shape and nullable historical metrics. Live rows populate LLM metrics
    # from the audit stream (FUP-093); #030 preserves older NULL payloads.

    # Deterministically computed from state.analysis_rounds.
    files_examined: int = Field(ge=0)
    # Union of resolved trace paths and fetched files outside the PR's changed
    # paths, not just Phase-2 fetches. See _compute_files_traced_beyond_diff.
    files_traced_beyond_diff: int = Field(ge=0)
    # aggregate_review_llm_metrics supplies the same audit-stream sums to
    # synthesize and the dashboard (FUP-093). None means an older unpopulated row,
    # not zero activity; replay must still accept it (#030).
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

    # Empty reports state the retained count and upstream risk with a fixed
    # template. Model-written prose can miscount, so render exact metrics from
    # metrics and the audit stream rather than parsing them from summary text.
    summary: str = Field(max_length=2000)
    overall_risk: RiskLevel
    # max_length is the HARD runaway ceiling (MAX_FINDINGS_HARD_CAP), NOT the soft
    # MAX_FINDINGS_PER_REPORT — protected findings survive the soft cap
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
