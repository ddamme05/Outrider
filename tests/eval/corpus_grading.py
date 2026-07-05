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
union: `expected_finding` rows assert a located (query, line) expectation — a
real vulnerability the catalog should reach, or, with
`real_vulnerability=false`, a documented non-vulnerability whose emission is a
tolerated false positive (carrying the outcome the catalog reaches TODAY plus,
for a non-emitted real vulnerability, the `residual_tag` naming the deferred
mechanism); `expected_clean` rows assert the catalog emits nothing for a file
(optionally scoped to one query id). `grade()` fail-louds before observing:
every vendored corpus file must carry at least one row (a row-less file would
never be parsed, so a catalog FP on it could ship unmeasured) and each row's
`query_match_id`/`finding_type` must resolve in the live registry AND be a
query production actually runs for the file's language. Observations key on
(query, line) with same-line multiplicity counts — line is the ground-truth
resolution, counts keep a 1 -> 2 emission drift visible. The grader
then compares the empirically-observed stage against each row's documented
current outcome and produces a deterministic `Scorecard` — the checked-in
`scorecard_juice_shop.json` pins current behavior, and the structural scenario
fails on any drift (a true positive regressing, a residual silently closing, or
a new false positive), forcing ground truth to be revisited.

No `tree_sitter` import crosses into this module: it consumes `parse_source`
output and the producer's domain records only (AST firewall).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Annotated, Literal, get_args
from unittest.mock import MagicMock
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from outrider.agent.nodes.analyze_observed import (
    module_admission_inputs_whole_file,
    produce_observed_findings,
    run_observed_matches,
)
from outrider.ast_facts.errors import UnsupportedExtensionError
from outrider.ast_facts.registry import parse_source
from outrider.coordinates import CoordinateError, query_span_to_source_lines
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.queries import registry

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from outrider.ast_facts.models import QueryMatchSpan

# A fixed review id so the harness is deterministic (findings' content hashes
# depend only on content, not this id, but keeping it constant makes any
# accidental id-dependence loud).
_GRADING_REVIEW_ID = UUID("00000000-0000-0000-0000-0000000c0d05")
_GRADING_INSTALLATION_ID = 1

# `denied_at_production` is structurally unreachable today — the producer emits
# every admitted match — and is retained only so stage attribution stays total
# if production ever grows a post-admission filter. No ground-truth row should
# pin it as a `current_outcome`.
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
# QueryAggregate's counter fields mirror this vocabulary; derive, don't copy.
_GRADE_FIELDS: tuple[str, ...] = get_args(Grade)


# ---------------------------------------------------------------------------
# Ground-truth row model (discriminated union on `kind`).
# ---------------------------------------------------------------------------
class ExpectedFindingRow(BaseModel):
    """A located (query, line) expectation. With `real_vulnerability=True`, a
    real vulnerability the catalog should reach; with `real_vulnerability=False`,
    a documented non-vulnerability pinning a tolerated false positive (an
    emission grades `false_positive`, never `true_positive`). `current_outcome`
    is what the catalog does today; when a real vulnerability's outcome is not
    `emitted`, `residual_tag` names the deferred mechanism (a declared
    non-goal) that explains the gap. `query_match_id`/`finding_type` are
    cross-checked against the live registry at grade time."""

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

    gradeable: bool
    # Why not, when gradeable is False (unreadable file, unsupported
    # extension, degraded parse) — surfaces in the not_graded rows' detail.
    ungraded_reason: str = ""
    # (query_match_id, line) -> occurrence count per layer. Line is the
    # ground-truth resolution (rows carry no span), but counts preserve
    # same-line multiplicity: two same-query emissions on one line must not
    # collapse, or a 1 -> 2 emission drift ships a byte-identical scorecard.
    raw: dict[tuple[str, int], int]
    admitted: dict[tuple[str, int], int]
    emitted: dict[tuple[str, int], int]


def _ungradeable(reason: str) -> _FileObservation:
    return _FileObservation(
        gradeable=False,
        ungraded_reason=reason,
        raw={},
        admitted={},
        emitted={},
    )


def _span_line(span: QueryMatchSpan, head_content: str) -> int | None:
    """Start line for a raw match span, under the producer's own guards: a
    zero-width envelope has no line range, and an unlocatable span cannot
    anchor a match. `run_observed_matches` SKIPS both (analyze_observed.py),
    so the raw layer must not crash where admission would skip — otherwise a
    degenerate span takes down the whole `grade()` aggregate."""
    if span.byte_end <= span.byte_start:
        return None
    try:
        line_start, _ = query_span_to_source_lines(
            byte_start=span.byte_start,
            byte_end=span.byte_end,
            head_content=head_content,
        )
    except CoordinateError:
        return None
    return line_start


def _observe_file(corpus_root: Path, rel_file: str) -> _FileObservation:
    """Run the three catalog layers over one corpus file, keyed by (query, line).

    The corpus grades a whole file (not a diff), so every extracted scope unit
    is `included` — the producer's scope-containment gate then sees the whole
    file as in-scope. Raw lines come from the canonical byte->line bridge;
    admitted/emitted lines from the producer's own records.

    Read/parse failures return an ungradeable observation (rows grade
    `not_graded`, never a crash and never a silent miss).
    """
    path = corpus_root / rel_file
    try:
        src = path.read_bytes()
    except OSError as exc:
        return _ungradeable(f"unreadable ({exc.__class__.__name__})")
    # Mirror production's byte frame: analyze parses and matches the
    # re-encoded DECODED content (intake hands the node a str), never the raw
    # on-disk bytes. On a non-UTF-8 file the U+FFFD re-encode shifts byte
    # offsets, so feeding every layer the same re-encoded bytes keeps the
    # (query, line) keys aligned across raw/admitted/emitted.
    head_content = src.decode("utf-8", errors="replace")
    data = head_content.encode("utf-8")
    try:
        parsed = parse_source(data, rel_file, MagicMock())
    except UnsupportedExtensionError:
        return _ungradeable("unsupported extension (no registered adapter)")
    if parsed.parser_outcome != "clean" or any(parsed.has_error.values()) or parsed.error_lines:
        # A degraded parse cannot be graded: production's `decide_degradation`
        # routes the file to a JUDGED-only degraded review, so OBSERVED
        # findings never reach the producer. JS/TS adapters report syntax
        # errors via `has_error`/`error_lines` while `parser_outcome` stays
        # "clean" (V1 pins it), so checking the outcome alone would grade a
        # broken file as if production reviewed it structurally.
        return _ungradeable("parse degraded (syntax errors)")

    language = registry.query_language_for_path(rel_file)
    grammar = registry.grammar_for_path(rel_file)
    raw: Counter[tuple[str, int]] = Counter()
    if language is not None and grammar is not None:
        for query_id in registry.observed_queries_for(language):
            for span in registry.match(query_id, data, grammar=grammar):
                line_start = _span_line(span, head_content)
                if line_start is not None:
                    raw[(query_id, line_start)] += 1

    module_scope_units, module_ranges = module_admission_inputs_whole_file(parsed, data)
    admitted_matches = run_observed_matches(
        file_path=rel_file,
        head_content=head_content,
        included_scope_units=parsed.scope_units,
        import_refs=parsed.imports,
        lexical_bindings=parsed.lexical_bindings,
        # The corpus grades a whole file as changed: the module-scope arm's
        # inputs derive through the whole-file sibling of the production
        # gate (DECISIONS.md#062) — the SAME error-free proof precondition,
        # with the entire file as the added range.
        all_scope_units=module_scope_units,
        added_line_ranges=module_ranges,
    )
    admitted = Counter((m.query_match_id, m.line_start) for m in admitted_matches)

    findings = produce_observed_findings(
        admitted_matches,
        file_path=rel_file,
        review_id=_GRADING_REVIEW_ID,
        installation_id=_GRADING_INSTALLATION_ID,
        active_policy_version=ACTIVE_POLICY_VERSION,
    )
    emitted = Counter((f.query_match_id, f.line_start) for f in findings)

    return _FileObservation(
        gradeable=True,
        raw=dict(raw),
        admitted=dict(admitted),
        emitted=dict(emitted),
    )


def _outcome_for(obs: _FileObservation, query_id: str, line: int) -> CorpusOutcome:
    """Attribute a (query, line) expectation to the deepest stage it reached
    (dict membership — counts don't change the stage, only the detail)."""
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


# Corpus-root artifacts that document the corpus rather than belong to it —
# excluded from the totality check (everything else must carry a row).
_CORPUS_METADATA_FILES = frozenset({"LICENSE", "MANIFEST.md", "ground_truth.json"})


def _corpus_source_files(corpus_root: Path) -> frozenset[str]:
    files: set[str] = set()
    for p in corpus_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(corpus_root).as_posix()
        if rel not in _CORPUS_METADATA_FILES:
            files.add(rel)
    return frozenset(files)


def _validate_corpus_totality(corpus_root: Path, ground_truth: GroundTruth) -> None:
    """Every vendored corpus file must carry at least one ground-truth row —
    `grade()` observes only files rows name, so a row-less file is never
    parsed and a catalog FP on it ships with a byte-identical scorecard. The
    reverse direction (a row naming a missing file, e.g. after a re-vendor
    rename) fails here with the pairing named instead of a FileNotFoundError
    mid-observation."""
    corpus_files = _corpus_source_files(corpus_root)
    row_files = {row.file for row in ground_truth.rows}
    uncovered = sorted(corpus_files - row_files)
    missing = sorted(row_files - corpus_files)
    problems: list[str] = []
    if uncovered:
        problems.append(f"corpus files with no ground-truth row: {uncovered}")
    if missing:
        problems.append(f"ground-truth rows naming missing files: {missing}")
    if problems:
        raise ValueError("corpus/ground-truth totality violated — " + "; ".join(problems))


def _validate_rows_against_registry(ground_truth: GroundTruth) -> None:
    """Fail loud on ground-truth/registry drift: a row naming an unknown query
    id would silently grade `no_raw_match`, a stale `finding_type` would
    survive a registry FindingType remap with a byte-identical scorecard, and
    a query production never runs for the file's language (a `.ts` row naming
    a `python.*` id) would grade as a plausible miss for a claim production
    cannot produce — all authoring/drift bugs to surface, not outcomes to
    grade."""
    problems: list[str] = []
    for row in ground_truth.rows:
        query_id = row.query_match_id
        if query_id is None:
            continue  # whole-catalog expected_clean rows carry no query scope
        observed_query = registry.OBSERVED_QUERIES.get(query_id)
        if observed_query is None:
            problems.append(f"{row.file}: unknown OBSERVED query id {query_id!r}")
            continue
        if (
            isinstance(row, ExpectedFindingRow)
            and row.finding_type != observed_query.finding_type.value
        ):
            problems.append(
                f"{row.file}: finding_type {row.finding_type!r} does not match the "
                f"registry's {observed_query.finding_type.value!r} for {query_id}"
            )
        file_language = registry.query_language_for_path(row.file)
        if query_id not in registry.observed_queries_for(file_language):
            problems.append(
                f"{row.file}: {query_id!r} is not a query production runs for this "
                f"file (query language {file_language!r})"
            )
    if problems:
        raise ValueError(
            "ground truth out of sync with the query registry — " + "; ".join(problems)
        )


def _validate_row_coherence(ground_truth: GroundTruth) -> None:
    """Reject duplicate and overlapping rows at load. Overlapping scopes
    double-grade regardless of the documented outcome (an emission counts as
    TP+FP today; a residual closing later grades one emission as both
    `improvement` and `false_positive`), so overlap is rejected structurally,
    not just for currently-emitted rows: a whole-catalog expected_clean row
    must be its file's ONLY row, and a query-scoped expected_clean row may not
    coexist with any expected_finding row for the same (file, query). The
    shipped corpus is coherent; this keeps future authoring honest."""
    problems: list[str] = []
    finding_keys: set[tuple[str, str, int]] = set()
    finding_scopes: set[tuple[str, str]] = set()
    clean_keys: set[tuple[str, str | None]] = set()
    rows_per_file: Counter[str] = Counter()
    for row in ground_truth.rows:
        rows_per_file[row.file] += 1
        if isinstance(row, ExpectedFindingRow):
            key = (row.file, row.query_match_id, row.line)
            if key in finding_keys:
                problems.append(f"duplicate expected_finding row {key}")
            finding_keys.add(key)
            finding_scopes.add((row.file, row.query_match_id))
        else:
            clean_key = (row.file, row.query_match_id)
            if clean_key in clean_keys:
                problems.append(f"duplicate expected_clean row {clean_key}")
            clean_keys.add(clean_key)
    for row in ground_truth.rows:
        if not isinstance(row, ExpectedCleanRow):
            continue
        if row.query_match_id is None:
            if rows_per_file[row.file] > 1:
                problems.append(
                    f"whole-catalog expected_clean row for {row.file} must be the "
                    f"file's only row ({rows_per_file[row.file]} rows present — "
                    f"overlapping scopes double-grade emissions)"
                )
        elif (row.file, row.query_match_id) in finding_scopes:
            problems.append(
                f"expected_clean scope ({row.file}, {row.query_match_id!r}) overlaps "
                f"expected_finding rows for the same query — one emission would "
                f"grade on both"
            )
    if problems:
        raise ValueError("ground truth rows are incoherent — " + "; ".join(problems))


def grade(ground_truth: GroundTruth, *, repo_root: Path) -> Scorecard:
    """Grade the catalog against ground truth, producing a deterministic scorecard."""
    corpus_root = repo_root / ground_truth.corpus_root
    _validate_rows_against_registry(ground_truth)
    _validate_row_coherence(ground_truth)
    _validate_corpus_totality(corpus_root, ground_truth)
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
            if not obs.gradeable:
                row_scores.append(
                    RowScore(
                        file=row.file,
                        query_match_id=row.query_match_id,
                        kind="expected_finding",
                        grade="not_graded",
                        observed_outcome=None,
                        current_outcome=row.current_outcome,
                        residual_tag=row.residual_tag,
                        detail=f"{obs.ungraded_reason} — not graded (never a silent miss)",
                    )
                )
                continue
            observed = _outcome_for(obs, row.query_match_id, row.line)
            grade_val, detail = _grade_expected_finding(row, observed)
            # Line is the ground-truth resolution, so all same-line emissions
            # of the row's query are claimed by this row — surface the count
            # so a 1 -> 2 multiplicity drift still moves the scorecard.
            emitted_count = obs.emitted.get((row.query_match_id, row.line), 0)
            if emitted_count > 1:
                detail += f" [{emitted_count} same-line emissions claimed by this row]"
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
            if not obs.gradeable:
                row_scores.append(
                    RowScore(
                        file=row.file,
                        query_match_id=row.query_match_id,
                        kind="expected_clean",
                        grade="not_graded",
                        observed_outcome=None,
                        current_outcome=None,
                        residual_tag=None,
                        detail=f"{obs.ungraded_reason} — not graded",
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
                multiple = [(k, obs.emitted[k]) for k in in_scope if obs.emitted[k] > 1]
                if multiple:
                    detail += f" [same-line multiplicities: {multiple}]"
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
    if not row.real_vulnerability:
        return _grade_documented_non_vulnerability(row, observed)
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


def _grade_documented_non_vulnerability(
    row: ExpectedFindingRow, observed: CorpusOutcome
) -> tuple[Grade, str]:
    """A row pinning catalog behavior on code that is NOT a real vulnerability.

    An emission here is a false positive — tolerated when documented
    (`current_outcome="emitted"`), new when not — and is never counted as a
    true positive, so the scorecard's TP/FP ledger stays honest. Non-emission
    on a non-vulnerability is the correct behavior regardless of the stage it
    stops at."""
    if observed == "emitted":
        if row.current_outcome == "emitted":
            return (
                "false_positive",
                "documented false positive (tolerated) — non-vulnerability emitted as pinned",
            )
        return (
            "false_positive",
            f"non-vulnerability now emitted (was '{row.current_outcome}') — new false positive",
        )
    if row.current_outcome == "emitted":
        return (
            "improvement",
            f"documented false positive no longer emitted (now '{observed}') — update ground truth",
        )
    if observed == row.current_outcome:
        return "true_negative", f"non-vulnerability correctly stopped at '{observed}'"
    return (
        "true_negative",
        f"non-vulnerability still not emitted; stage drifted "
        f"'{row.current_outcome}' -> '{observed}' — update ground truth",
    )


def _unclaimed_emissions(
    observations: dict[str, _FileObservation], claimed: set[tuple[str, str, int]]
) -> list[RowScore]:
    out: list[RowScore] = []
    for file, obs in observations.items():
        for qid, line in sorted(obs.emitted):
            if (file, qid, line) not in claimed:
                count = obs.emitted[(qid, line)]
                multiplicity = f" [{count} emissions]" if count > 1 else ""
                out.append(
                    RowScore(
                        file=file,
                        query_match_id=qid,
                        kind="expected_clean",
                        grade="unexpected_emission",
                        observed_outcome="emitted",
                        current_outcome=None,
                        residual_tag=None,
                        detail=(
                            f"emitted at line {line} with no ground-truth row "
                            f"(unlabeled){multiplicity}"
                        ),
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
