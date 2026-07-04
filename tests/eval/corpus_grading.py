"""Grade the JS/TS OBSERVED catalog against the vendored Juice Shop corpus.

The measurement arc's engine (`specs/2026-07-04-juice-shop-graded-corpus.md`).
LLM-free and DB-free: for each corpus file it drives the same three layers the
production analyze path does — `registry.match` (raw structural match),
`run_observed_matches` (producer admission), `produce_observed_findings`
(emitted findings) — and attributes every graded outcome to the STAGE it
reached, so a real vulnerability the catalog drops is separable into
"the query never matched" vs "admission denied it" (a binding / shadowing /
module_presence residual) vs "a finding was emitted".

Ground truth (`corpus/juice_shop/ground_truth.json`) is a discriminated row
union: `expected_finding` rows assert a real vulnerability the catalog should
reach (carrying the outcome it reaches TODAY plus, when that is not `emitted`,
the `residual_tag` naming the deferred mechanism); `expected_clean` rows assert
the catalog emits nothing for a file (optionally scoped to one query id). The
grader compares the empirically-observed stage against each row's documented
current outcome and produces a deterministic `Scorecard` — the checked-in
`scorecard_juice_shop.json` pins current behavior, and the structural scenario
fails on any drift (a true positive regressing, a residual silently closing, or
a new false positive), forcing ground truth to be revisited.

No `tree_sitter` import crosses into this module: it consumes `parse_source`
output and the producer's domain records only (AST firewall).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Literal
from unittest.mock import MagicMock
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from outrider.agent.nodes.analyze_observed import (
    produce_observed_findings,
    run_observed_matches,
)
from outrider.ast_facts.registry import parse_source
from outrider.coordinates import query_span_to_source_lines
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.queries import registry

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# A fixed review id so the harness is deterministic (findings' content hashes
# depend only on content, not this id, but keeping it constant makes any
# accidental id-dependence loud).
_GRADING_REVIEW_ID = UUID("00000000-0000-0000-0000-0000000c0d05")
_GRADING_INSTALLATION_ID = 1

CorpusOutcome = Literal["emitted", "denied_at_production", "denied_at_admission", "no_raw_match"]
Grade = Literal[
    "true_positive",
    "accepted_miss",
    "true_negative",
    "false_positive",
    "regression",
    "improvement",
    "unexpected_emission",
    "not_graded",
]
_GRADE_FIELDS: tuple[str, ...] = (
    "true_positive",
    "accepted_miss",
    "true_negative",
    "false_positive",
    "regression",
    "improvement",
    "unexpected_emission",
    "not_graded",
)


# ---------------------------------------------------------------------------
# Ground-truth row model (discriminated union on `kind`).
# ---------------------------------------------------------------------------
class ExpectedFindingRow(BaseModel):
    """A real vulnerability the catalog should reach. `current_outcome` is what
    the catalog does today; when it is not `emitted`, `residual_tag` names the
    deferred mechanism (a declared non-goal) that explains the gap."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["expected_finding"]
    file: str
    query_match_id: str
    finding_type: str
    line: int
    real_vulnerability: bool
    current_outcome: CorpusOutcome
    residual_tag: str | None = None
    rationale: str


class ExpectedCleanRow(BaseModel):
    """An assertion that the catalog emits NOTHING for `file` — optionally
    scoped to one `query_match_id` (None = the whole catalog must be silent).
    An absence expectation carries no location: no line/span/finding_type."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["expected_clean"]
    file: str
    query_match_id: str | None = None
    rationale: str


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    comment: str | None = Field(default=None, alias="_comment")
    corpus_root: str
    rows: list[Annotated[ExpectedFindingRow | ExpectedCleanRow, Field(discriminator="kind")]]


# ---------------------------------------------------------------------------
# Scorecard model (the checked-in artifact).
# ---------------------------------------------------------------------------
class RowScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    query_match_id: str | None
    kind: Literal["expected_finding", "expected_clean"]
    grade: Grade
    observed_outcome: CorpusOutcome | None
    current_outcome: CorpusOutcome | None
    residual_tag: str | None
    detail: str


class QueryAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_match_id: str
    true_positive: int = 0
    accepted_miss: int = 0
    true_negative: int = 0
    false_positive: int = 0
    regression: int = 0
    improvement: int = 0
    unexpected_emission: int = 0
    not_graded: int = 0


class Scorecard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_root: str
    row_scores: list[RowScore]
    by_query: list[QueryAggregate]
    totals: QueryAggregate

    def to_json(self) -> str:
        """Deterministic serialization for the checked-in artifact: sorted keys,
        2-space indent, trailing newline. No timestamps (would break equality)."""
        return json.dumps(self.model_dump(), sort_keys=True, indent=2) + "\n"


# ---------------------------------------------------------------------------
# The three-layer observation over one file.
# ---------------------------------------------------------------------------
class _FileObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parsed_clean: bool
    # (query_match_id, line) tuples for each layer.
    raw: frozenset[tuple[str, int]]
    admitted: frozenset[tuple[str, int]]
    emitted: frozenset[tuple[str, int]]


def _observe_file(corpus_root: Path, rel_file: str) -> _FileObservation:
    """Run the three catalog layers over one corpus file, keyed by (query, line).

    The corpus grades a whole file (not a diff), so every extracted scope unit
    is `included` — the producer's scope-containment gate then sees the whole
    file as in-scope. Raw lines come from the canonical byte->line bridge;
    admitted/emitted lines from the producer's own records.
    """
    path = corpus_root / rel_file
    src = path.read_bytes()
    parsed = parse_source(src, rel_file, MagicMock())
    if parsed.parser_outcome != "clean":
        # A degraded parse cannot be graded (findings never reach the producer);
        # mark it not-clean so the caller grades its rows `not_graded`, never a
        # silent miss.
        return _FileObservation(
            parsed_clean=False, raw=frozenset(), admitted=frozenset(), emitted=frozenset()
        )

    head_content = src.decode("utf-8", errors="replace")
    language = registry.query_language_for_path(rel_file)
    grammar = registry.grammar_for_path(rel_file)
    raw: set[tuple[str, int]] = set()
    if language is not None and grammar is not None:
        for query_id in registry.observed_queries_for(language):
            for span in registry.match(query_id, src, grammar=grammar):
                line_start, _ = query_span_to_source_lines(
                    byte_start=span.byte_start,
                    byte_end=span.byte_end,
                    head_content=head_content,
                )
                raw.add((query_id, line_start))

    admitted_matches = run_observed_matches(
        file_path=rel_file,
        head_content=head_content,
        included_scope_units=parsed.scope_units,
        import_refs=parsed.imports,
        lexical_bindings=parsed.lexical_bindings,
    )
    admitted = {(m.query_match_id, m.line_start) for m in admitted_matches}

    findings = produce_observed_findings(
        admitted_matches,
        file_path=rel_file,
        review_id=_GRADING_REVIEW_ID,
        installation_id=_GRADING_INSTALLATION_ID,
        active_policy_version=ACTIVE_POLICY_VERSION,
    )
    emitted = {(f.query_match_id, f.line_start) for f in findings}

    return _FileObservation(
        parsed_clean=True,
        raw=frozenset(raw),
        admitted=frozenset(admitted),
        emitted=frozenset(emitted),
    )


def _outcome_for(obs: _FileObservation, query_id: str, line: int) -> CorpusOutcome:
    """Attribute a (query, line) expectation to the deepest stage it reached."""
    if (query_id, line) in obs.emitted:
        return "emitted"
    if (query_id, line) in obs.admitted:
        return "denied_at_production"
    if (query_id, line) in obs.raw:
        return "denied_at_admission"
    return "no_raw_match"


# ---------------------------------------------------------------------------
# Grading.
# ---------------------------------------------------------------------------
def load_ground_truth(path: Path) -> GroundTruth:
    return GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))


def grade(ground_truth: GroundTruth, *, repo_root: Path) -> Scorecard:
    """Grade the catalog against ground truth, producing a deterministic scorecard."""
    corpus_root = repo_root / ground_truth.corpus_root
    # Observe each referenced file once.
    files = sorted({row.file for row in ground_truth.rows})
    observations = {f: _observe_file(corpus_root, f) for f in files}

    row_scores: list[RowScore] = []
    # Track which (file, query, line) emissions were "claimed" by an
    # expected_finding row, so leftover emissions on those files can be flagged
    # as unexpected (a new FP not yet labeled).
    claimed: set[tuple[str, str, int]] = set()

    for row in ground_truth.rows:
        if isinstance(row, ExpectedFindingRow):
            obs = observations[row.file]
            claimed.add((row.file, row.query_match_id, row.line))
            if not obs.parsed_clean:
                row_scores.append(
                    RowScore(
                        file=row.file,
                        query_match_id=row.query_match_id,
                        kind="expected_finding",
                        grade="not_graded",
                        observed_outcome=None,
                        current_outcome=row.current_outcome,
                        residual_tag=row.residual_tag,
                        detail="parse degraded — not graded (never a silent miss)",
                    )
                )
                continue
            observed = _outcome_for(obs, row.query_match_id, row.line)
            grade_val, detail = _grade_expected_finding(row, observed)
            row_scores.append(
                RowScore(
                    file=row.file,
                    query_match_id=row.query_match_id,
                    kind="expected_finding",
                    grade=grade_val,
                    observed_outcome=observed,
                    current_outcome=row.current_outcome,
                    residual_tag=row.residual_tag,
                    detail=detail,
                )
            )
        else:
            obs = observations[row.file]
            if not obs.parsed_clean:
                row_scores.append(
                    RowScore(
                        file=row.file,
                        query_match_id=row.query_match_id,
                        kind="expected_clean",
                        grade="not_graded",
                        observed_outcome=None,
                        current_outcome=None,
                        residual_tag=None,
                        detail="parse degraded — not graded",
                    )
                )
                continue
            in_scope = [
                (qid, line)
                for (qid, line) in sorted(obs.emitted)
                if row.query_match_id is None or qid == row.query_match_id
            ]
            if in_scope:
                grade_val = "false_positive"
                detail = f"emitted {in_scope} but file is expected clean"
            else:
                grade_val = "true_negative"
                detail = "no findings emitted (correct)"
            for qid, line in in_scope:
                claimed.add((row.file, qid, line))
            row_scores.append(
                RowScore(
                    file=row.file,
                    query_match_id=row.query_match_id,
                    kind="expected_clean",
                    grade=grade_val,
                    observed_outcome=None,
                    current_outcome=None,
                    residual_tag=None,
                    detail=detail,
                )
            )

    # Any emission on a graded file not claimed by a row is an unexpected FP.
    row_scores.extend(_unclaimed_emissions(observations, claimed))

    row_scores.sort(key=lambda r: (r.file, r.query_match_id or "", r.kind, r.grade))
    by_query = _aggregate_by_query(row_scores)
    totals = _aggregate_total(by_query)
    return Scorecard(
        corpus_root=ground_truth.corpus_root,
        row_scores=row_scores,
        by_query=by_query,
        totals=totals,
    )


def _grade_expected_finding(row: ExpectedFindingRow, observed: CorpusOutcome) -> tuple[Grade, str]:
    if observed == row.current_outcome:
        if observed == "emitted":
            return "true_positive", "emitted as expected"
        return (
            "accepted_miss",
            f"denied at '{observed}' as documented (residual: {row.residual_tag})",
        )
    # Drift from the documented current behavior.
    if observed == "emitted":
        return (
            "improvement",
            f"now emitted (was '{row.current_outcome}', residual "
            f"'{row.residual_tag}' may have closed — update ground truth)",
        )
    if row.current_outcome == "emitted":
        return (
            "regression",
            f"expected emitted, observed '{observed}' — a true positive regressed",
        )
    return (
        "regression",
        f"expected '{row.current_outcome}', observed '{observed}' — stage drift",
    )


def _unclaimed_emissions(
    observations: dict[str, _FileObservation], claimed: set[tuple[str, str, int]]
) -> list[RowScore]:
    out: list[RowScore] = []
    for file, obs in observations.items():
        for qid, line in sorted(obs.emitted):
            if (file, qid, line) not in claimed:
                out.append(
                    RowScore(
                        file=file,
                        query_match_id=qid,
                        kind="expected_clean",
                        grade="unexpected_emission",
                        observed_outcome="emitted",
                        current_outcome=None,
                        residual_tag=None,
                        detail=f"emitted at line {line} with no ground-truth row (unlabeled)",
                    )
                )
    return out


def _aggregate_by_query(row_scores: Iterable[RowScore]) -> list[QueryAggregate]:
    buckets: dict[str, QueryAggregate] = {}
    for r in row_scores:
        qid = r.query_match_id or "(catalog-wide)"
        agg = buckets.setdefault(qid, QueryAggregate(query_match_id=qid))
        setattr(agg, r.grade, getattr(agg, r.grade) + 1)
    return [buckets[k] for k in sorted(buckets)]


def _aggregate_total(by_query: Iterable[QueryAggregate]) -> QueryAggregate:
    total = QueryAggregate(query_match_id="(total)")
    for agg in by_query:
        for field in _GRADE_FIELDS:
            setattr(total, field, getattr(total, field) + getattr(agg, field))
    return total
