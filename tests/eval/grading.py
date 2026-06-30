"""Deterministic finding-grading for the analyze model-tier quality gate.

This is the load-bearing methodology for `specs/2026-06-08-analyze-tiered-model-routing.md`
step 2 (the eval quality gate) — it gated the STANDARD→Haiku flip (`DECISIONS.md#041`, now
shipped): before STANDARD-tier files could move from Sonnet to a cheaper model, we needed
DETERMINISTIC evidence the cheaper model did not lose the known findings. So grading is
STRUCTURAL, not LLM-as-judge — we do not put a model in the loop
that is meant to validate a model (that would replace "did Haiku preserve recall?" with
"does another model think Haiku preserved recall?").

A model finding MATCHES a ground-truth finding iff ALL hold (the declared match contract):
  1. same `finding_type`,
  2. same `file_path`,
  3. the finding's line range overlaps the expected range expanded by `line_window`,
  4. same `severity` (policy-derived — a belt-and-suspenders check that catches a policy
     drift even though severity follows `finding_type`).

From the match set we compute recall (of the expected findings, how many were caught),
precision (of the model's findings, how many matched an expected one — so extra/noise
findings are counted separately, not hidden), and severity accuracy. `compare(...)` then
applies the gate, which passes only if ALL THREE hold: the BASELINE itself cleared a
DECLARED recall floor (`baseline_valid` — else a scenario both models fail would "hold
recall" vacuously), the candidate's recall is within a DECLARED tolerance of the
baseline's, AND its false-positive (extra) count did not balloon.

Pure functions over already-validated `ReviewFinding`s + `ExpectedFinding` ground truth;
no I/O, no LLM, no spend. The real-model run that produces the `ReviewFinding`s lives in
the opt-in comparison runner; this module only grades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from outrider.policy import (  # noqa: TC001 — runtime use: Pydantic resolves these field annotations
    FindingSeverity,
    FindingType,
)

from .metrics import FindingPrecision, FindingRecall, SeverityAccuracy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.schemas.review_finding import ReviewFinding

# Default line-distance window: a finding whose range is within this many lines of the
# expected range (after overlap expansion) still matches. Small + declared — a model that
# points at the right vulnerability a line or two off is the same finding, not a miss; a
# model that points at a different function is not.
DEFAULT_LINE_WINDOW = 2


class ExpectedFinding(BaseModel):
    """One ground-truth finding a scenario is KNOWN to contain. The grader matches a
    model's `ReviewFinding`s against a tuple of these. Frozen — ground truth is fixed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_path: str
    line_start: int
    line_end: int
    finding_type: FindingType
    severity: FindingSeverity


def _line_ranges_match(
    *, actual_start: int, actual_end: int, expected_start: int, expected_end: int, line_window: int
) -> bool:
    """True if `[actual_start, actual_end]` overlaps `[expected_start, expected_end]`
    expanded by `line_window` on each side. Overlap of `[a, b]` and `[c, d]` is
    `a <= d and c <= b`; the window widens the expected interval."""
    lo = expected_start - line_window
    hi = expected_end + line_window
    return actual_start <= hi and lo <= actual_end


def finding_matches(
    actual: ReviewFinding, expected: ExpectedFinding, *, line_window: int = DEFAULT_LINE_WINDOW
) -> bool:
    """Whether one model finding matches one ground-truth finding (the declared contract:
    same type + same file + line overlap-within-window + same severity)."""
    return (
        actual.finding_type == expected.finding_type
        and actual.file_path == expected.file_path
        and actual.severity == expected.severity
        and _line_ranges_match(
            actual_start=actual.line_start,
            actual_end=actual.line_end,
            expected_start=expected.line_start,
            expected_end=expected.line_end,
            line_window=line_window,
        )
    )


@dataclass(frozen=True, slots=True)
class GradeResult:
    """One model's grade against a scenario's ground truth.

    `recall`/`precision`/`severity_accuracy` are the `metrics.py` ratio shapes.
    `missed` are expected findings no model finding matched (the recall losses — the
    thing this gate exists to catch). `extra` are model findings that matched NO expected
    finding (the false positives — counted separately so noise can't hide)."""

    recall: FindingRecall
    precision: FindingPrecision
    severity_accuracy: SeverityAccuracy
    missed: tuple[ExpectedFinding, ...]
    extra: tuple[ReviewFinding, ...]
    n_matched: int

    @property
    def n_false_positives(self) -> int:
        return len(self.extra)


def grade(
    actual: Sequence[ReviewFinding],
    expected: Sequence[ExpectedFinding],
    *,
    line_window: int = DEFAULT_LINE_WINDOW,
) -> GradeResult:
    """Grade one model's findings against ground truth (structural match).

    Each expected finding is matched by AT MOST one actual finding (greedy first-match);
    each actual finding can satisfy at most one expected finding, so two model findings
    can't both "claim" one ground-truth finding to inflate recall. Severity accuracy is
    over the MATCHED set (of the findings that matched type+file+line, how many also got
    severity right — though the match contract already requires it, so this is 1.0 unless
    the contract is loosened; kept explicit for when the window/severity rules diverge).
    """
    expected_list = list(expected)
    matched_expected_idx: set[int] = set()
    # Greedy one-to-one matching: each actual finding claims at most one not-yet-claimed
    # expected finding, recorded as a PAIR so downstream metrics score over the actual
    # pairing rather than a cross product.
    matched_pairs: list[tuple[ReviewFinding, ExpectedFinding]] = []
    extra: list[ReviewFinding] = []

    for af in actual:
        hit_idx: int | None = None
        for i, ef in enumerate(expected_list):
            if i in matched_expected_idx:
                continue
            if finding_matches(af, ef, line_window=line_window):
                hit_idx = i
                break
        if hit_idx is None:
            extra.append(af)
        else:
            matched_expected_idx.add(hit_idx)
            matched_pairs.append((af, expected_list[hit_idx]))

    n_expected = len(expected_list)
    n_matched = len(matched_pairs)
    # From the partition (each actual is matched xor extra) — not a re-iteration of
    # `actual`, which would miscount a one-shot iterable as 0 and force precision to 1.0.
    n_actual = n_matched + len(extra)
    missed = tuple(ef for i, ef in enumerate(expected_list) if i not in matched_expected_idx)

    recall = FindingRecall(
        value=(n_matched / n_expected) if n_expected else 1.0,
        numerator=n_matched,
        denominator=n_expected,
    )
    precision = FindingPrecision(
        value=(n_matched / n_actual) if n_actual else 1.0,
        numerator=n_matched,
        denominator=n_actual,
    )
    # Of the matched findings, how many also got severity right — scored over the greedy
    # PAIRS (each actual with its one matched expected), not a cross product. The match
    # contract already requires severity equality, so this is 1.0 today; it stays a separate
    # metric so a future looser match rule (match on type+line, score severity separately)
    # needs no new plumbing.
    n_sev_ok = sum(1 for af, ef in matched_pairs if af.severity == ef.severity)
    severity_accuracy = SeverityAccuracy(
        value=(n_sev_ok / n_matched) if n_matched else 1.0,
        numerator=n_sev_ok,
        denominator=n_matched,
    )

    return GradeResult(
        recall=recall,
        precision=precision,
        severity_accuracy=severity_accuracy,
        missed=missed,
        extra=tuple(extra),
        n_matched=n_matched,
    )


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """Baseline (Sonnet) vs candidate (Haiku) on one scenario, with the gate verdict.

    Three conditions, all DECLARED-threshold and auditable:
      - `baseline_valid` — the BASELINE itself cleared `baseline_recall_floor`. This guards
        the VACUOUS pass: if the strong model didn't catch the scenario's known findings,
        the scenario can't discriminate, and `recall_held` is trivially true (candidate
        0.0 >= baseline 0.0). A scenario the baseline can't solve certifies nothing.
      - `recall_held` — the candidate's recall is within `recall_tolerance` of the baseline.
      - `fp_bounded` — the candidate's false positives did not exceed the baseline's by more
        than `fp_allowance` (a RELATIVE bound: candidate FP <= baseline FP + `fp_allowance`).
    `passes` is the AND of all three: a recall hold over a non-discriminating baseline does
    NOT pass."""

    baseline: GradeResult
    candidate: GradeResult
    recall_tolerance: float
    fp_allowance: int
    baseline_recall_floor: float
    baseline_valid: bool
    recall_held: bool
    fp_bounded: bool
    # Structured-output yield (FUP-196): did the model emit PARSEABLE structured output, or
    # was its response REJECTED (fence/schema fail → zero findings)? Distinguishes a
    # capability miss (valid output, no finding) from a format miss (rejected) — the two are
    # otherwise indistinguishable to recall/precision on safe code (both → empty findings).
    baseline_rejected: bool = False
    candidate_rejected: bool = False

    @property
    def passes(self) -> bool:
        return self.baseline_valid and self.recall_held and self.fp_bounded


def compare(
    baseline: GradeResult,
    candidate: GradeResult,
    *,
    recall_tolerance: float = 0.0,
    fp_allowance: int = 0,
    baseline_recall_floor: float = 1.0,
    baseline_rejected: bool = False,
    candidate_rejected: bool = False,
) -> ModelComparison:
    """Apply the quality gate. The candidate (cheaper) model passes iff ALL hold:
      - the BASELINE's own recall clears `baseline_recall_floor` (default 1.0 — the strong
        model must catch the scenario's known findings, else "recall held" is vacuous: a
        scenario both models fail can't certify the candidate);
      - the candidate's recall is within `recall_tolerance` of the baseline's; AND
      - it added no more than `fp_allowance` false positives over the baseline.
    Defaults are STRICT (baseline perfect, no recall loss, no extra FPs) — callers loosen
    them explicitly and the chosen values are recorded on the result."""
    baseline_valid = baseline.recall.value >= baseline_recall_floor
    recall_held = candidate.recall.value >= baseline.recall.value - recall_tolerance
    fp_bounded = candidate.n_false_positives <= baseline.n_false_positives + fp_allowance
    return ModelComparison(
        baseline=baseline,
        candidate=candidate,
        recall_tolerance=recall_tolerance,
        fp_allowance=fp_allowance,
        baseline_recall_floor=baseline_recall_floor,
        baseline_valid=baseline_valid,
        recall_held=recall_held,
        fp_bounded=fp_bounded,
        baseline_rejected=baseline_rejected,
        candidate_rejected=candidate_rejected,
    )
