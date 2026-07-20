"""Arc 2 — strict-schema feasibility probe (the arc's ONLY entrypoint).

Answers two unverified questions with ONE bounded paid capture
(specs/2026-07-20-arc2-strict-schema-feasibility.md):

  1. Does the live API accept Outrider's exact proof-preserving,
     evidence-tier-discriminated strict `json_schema`?
  2. Does GPT-5.6 populate the API-owned `message.refusal` channel for it —
     the discriminator the production `json_object` shape does not have, and
     the reason the whole `openai` host is production-unadmitted
     (`DECISIONS.md#056`, 2026-07-19 amendment / FUP-246 trigger 2)?

**Gate 1 (this change): there is no paid runner.** `run_paid()` is a stub that
refuses unconditionally — it accepts no contract, loads none, and checks nothing.
Nothing here can spend, because nothing here can call the API. `--dry-run` and
`--self-test` are free and are the only things that run today.

Stated that plainly on purpose. An earlier version of this paragraph described
the lock as "a missing type" that `run_paid()` requires, which read as though the
function performed a contract check; it does not, and a reader would have gone
looking for one. The contract machinery is a Gate-2 REQUIREMENT, not a Gate-1
mechanism.

**Gate 2 (separate, reviewed):** the paid runner lands, and MUST obtain a
`PaidProbeContract` rather than reading caps from constants. The caps MEASURED by
`--dry-run` are reviewed by a human and passed to `from_reviewed` (a per-row
prompt-byte cap and a `max_completion_tokens` sent under the profile-declared
`token_limit_param`, which SHAPER v3 established is `max_completion_tokens` for
GPT-5.6). Caps are generation-affecting, so the contract binds them into identity
— see `arc2/contracts.py`, and `arc2/verifier.py` for how a capture is replayed
against that identity.

Deliberately NOT invented up front: numeric caps before the real request bodies
exist would be conformance theater, and two implementations could spend
materially differently while both claiming to follow the spec.

Run (FREE):
    uv run --no-sync python spikes/openai/strict_schema_probe.py --dry-run
    uv run --no-sync python spikes/openai/strict_schema_probe.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable  # noqa: TC003 - runtime annotation in _coherence_cases
from pathlib import Path
from typing import Any, Final

from outrider.llm.host_profiles import OPENAI_PROFILE

# Namespace-package bootstrap so `spikes.openai.arc2...` resolves when this file is
# run as a script. Guarded so importing this module does not append a DUPLICATE
# entry: `sys.path` is process-global, and a second copy survives a caller's single
# `sys.path.remove()` — which is exactly how `tests/unit/conftest.py` was silently
# leaving the repo root importable for the whole pytest process.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from spikes.openai.arc2.classifier import (  # noqa: E402 - after sys.path bootstrap
    ROW_ORDER,
    EvaluatedRow,
    FindingRowAssessment,
    ObservationKind,
    ParserOutcome,
    RowId,
    Verdict,
    classify_envelope,
    classify_session,
    classify_transport,
    evaluate_row,
    route_finding_assessment,
)
from spikes.openai.arc2.contracts import (  # noqa: E402 - after sys.path bootstrap
    STRICT_PROBE_CONTRACT_VERSION,
    DryRunManifest,
)

# Re-exported for the CLI and the offline tests. These are ALIASES for the plan's
# objects, never copies: `plan.py` is the single authority for both the evaluation
# identity and the generation identity, and this module is the runner over it.
from spikes.openai.arc2.plan import (  # noqa: E402 - after sys.path bootstrap
    CLEAN_SCENARIO,
    DEFECT_SCENARIO,
    EXPECTED_FINDING,
    FIRED_QUERY_MATCH_IDS,
    MAX_COMPLETION_TOKENS,
    MODEL,
    Scenario,
    _measure,
    analyze_prompt_pair,
    build_rows,
    derive_fired_query_match_ids,
    production_kwargs,
    refusal_freeze_error,
    registered_evaluation_contract,
    registered_locked_contract,
    registered_measurements,
    serialize_kwargs,
    strict_kwargs,
    strict_response_format,
)
from spikes.openai.arc2.strict_schema import (  # noqa: E402 - after sys.path bootstrap
    derive_strict_analyze_schema,
    schema_digest,
    strict_schema_json,
)

__all__ = [
    "CLEAN_SCENARIO",
    "DEFECT_SCENARIO",
    "EXPECTED_FINDING",
    "FIRED_QUERY_MATCH_IDS",
    "MAX_COMPLETION_TOKENS",
    "MODEL",
    "Scenario",
    "analyze_prompt_pair",
    "build_rows",
    "derive_fired_query_match_ids",
    "main",
    "production_kwargs",
    "refusal_freeze_error",
    "registered_evaluation_contract",
    "registered_locked_contract",
    "registered_measurements",
    "run_dry",
    "run_paid",
    "run_real_parser",
    "run_self_test",
    "serialize_kwargs",
    "strict_kwargs",
    "strict_response_format",
]

_OUT_DIR: Final[Path] = Path(__file__).resolve().parent / "arc2_out"


def run_dry() -> int:
    """FREE. Derive the schema, build every row, measure, write the artifact.

    Emits a content-addressed `DryRunManifest` when the refusal prompts are
    FROZEN, and an explicitly `preview_only` file when they are still
    placeholders. A preview is mechanically ineligible as a Gate-2 source: it
    measures prompts that will not be the ones sent, so caps derived from it
    would describe a different experiment.
    """
    schema = derive_strict_analyze_schema()
    rows = build_rows()
    measurements = tuple(_measure(row_id, rows[row_id]) for row_id in ROW_ORDER)
    freeze_error = refusal_freeze_error()

    print(f"contract          : {STRICT_PROBE_CONTRACT_VERSION}")
    print(f"model             : {MODEL}")
    print(f"base_url          : {OPENAI_PROFILE.base_url}")
    print(f"profile digest    : {OPENAI_PROFILE.profile_contract_digest[:16]}")
    schema_bytes = len(strict_schema_json(schema))
    print(f"strict schema     : {schema_digest(schema)[:16]}  ({schema_bytes} B)")
    print(f"branches          : {len(schema['properties']['findings']['items']['anyOf'])}")
    print(f"completion limit  : {MAX_COMPLETION_TOKENS} (request-effective)")
    print(f"artifact          : {'PREVIEW ONLY' if freeze_error else 'Gate-2 eligible'}")
    print()
    print(f"{'row':<22} {'prompt bytes':>13} {'est tokens':>11}  body digest")
    for m in measurements:
        print(
            f"{m.row_id.value:<22} {m.prompt_bytes:>13,} {m.estimated_tokens:>11,}  "
            f"{m.request_body_digest[:16]}"
        )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    if freeze_error is not None:
        out = _OUT_DIR / "dry_run_preview_only.json"
        out.write_text(
            json.dumps(
                {
                    "preview_only": True,
                    "ineligible_reason": freeze_error,
                    "contract_version": STRICT_PROBE_CONTRACT_VERSION,
                    "measurements": [m.model_dump(mode="json") for m in measurements],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"preview artifact  : {out}")
        print(f"NOT Gate-2 eligible: {freeze_error}")
        return 0

    locked = registered_locked_contract()
    manifest = DryRunManifest(
        contract_version=STRICT_PROBE_CONTRACT_VERSION,
        locked_contract_digest=locked.digest,
        measurements=measurements,
    )
    # Content-addressed: a re-measurement under different inputs writes a
    # DIFFERENT file rather than overwriting the reviewed one.
    out = _OUT_DIR / f"dry_run_{manifest.digest[:16]}.json"
    if out.exists():
        print(f"dry-run artifact  : {out} (unchanged)")
    else:
        out.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", "utf-8")
        print(f"dry-run artifact  : {out}")
    print(f"locked digest     : {locked.digest[:16]}")
    print(f"manifest digest   : {manifest.digest[:16]}")
    print()
    print(
        "Gate 2 input. Review these MEASURED numbers, then construct a "
        "PaidProbeContract via `PaidProbeContract.from_reviewed(locked, dry, caps)` "
        "— caps are the only new data it accepts, and identity is copied from the "
        "verified parents."
    )
    return 0


# ---------------------------------------------------------------------------
# `--self-test` — every terminal verdict exercised from recorded fixtures.
# ---------------------------------------------------------------------------


def run_real_parser(body: str) -> ParserOutcome:
    """Report what the REAL `parse_analyze_response` observed for this body.

    DELEGATES to `ParserAdapter.for_registered_plan()` — the same adapter replay
    uses — rather than assembling the call itself. It used to duplicate that
    assembly and omit `pass_index` / `trace_candidate_form`, so the offline
    self-test could exercise different admission inputs from the ones replay grades
    under. Two parser call sites is two grader identities; there is now one.
    """
    from spikes.openai.arc2.verifier import ParserAdapter

    return ParserAdapter.for_registered_plan()(body)


_EMPTY_BODY: Final[str] = '{"findings":[]}'


def _finding_body(**overrides: Any) -> str:
    """A REAL one-finding body at the scenario's OWN coordinates.

    Self-test rows are built from bodies, never from a bare assessment:
    `EvaluatedRow` derives every semantic flag, so an "impossible observation"
    fixture cannot be written.
    """
    finding: dict[str, Any] = {
        "finding_type": "sql_injection",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "SQL built by concatenation",
        "description": "User input is concatenated into a SQL string.",
        "evidence": "conn.execute(...)",
        # The scenario authority's coordinates — NOT hand-picked. Hand-authored
        # line numbers agreed with a stale expectation instead of with the file,
        # which is what concealed the three-frame disagreement.
        "line_start": DEFECT_SCENARIO.defect_line,
        "line_end": DEFECT_SCENARIO.defect_line,
        "trace_candidates": [],
    }
    finding.update(overrides)
    return json.dumps({"findings": [finding]})


def _row(
    row: RowId,
    *,
    content: str | None = None,
    refusal: str | None = None,
    parser: Callable[[str], ParserOutcome] | None = None,
) -> EvaluatedRow:
    """Build an `EvaluatedRow` from bytes. Every semantic flag is DERIVED from
    those bytes plus the parser's facts — the self-test cannot hand-supply one."""
    return evaluate_row(
        row=row,
        content=content,
        refusal=refusal,
        finish_reason="stop",
        run_parser=parser,
        fired_query_match_ids=FIRED_QUERY_MATCH_IDS,
        expected_finding=EXPECTED_FINDING,
    )


def _session(
    *,
    refusals: dict[RowId, EvaluatedRow],
    finding: EvaluatedRow | None = None,
    clean: EvaluatedRow | None = None,
) -> Verdict:
    rows: dict[RowId, EvaluatedRow] = {
        RowId.ACCEPTANCE_CLEAN: clean
        if clean is not None
        else _row(RowId.ACCEPTANCE_CLEAN, content=_EMPTY_BODY),
        RowId.ACCEPTANCE_FINDING: finding
        if finding is not None
        else _row(RowId.ACCEPTANCE_FINDING, content=_finding_body(), parser=run_real_parser),
    }
    rows.update(refusals)
    return classify_session(rows=rows).verdict


def _self_test_cases() -> list[tuple[str, Verdict, Verdict]]:
    """(name, expected, actual) for every terminal verdict + both feeder kinds."""
    all_negative = {
        row: _row(row, content=_EMPTY_BODY)
        for row in (RowId.REFUSAL_1, RowId.REFUSAL_2, RowId.REFUSAL_3)
    }
    refused_second = dict(all_negative)
    refused_second[RowId.REFUSAL_2] = _row(RowId.REFUSAL_2, refusal="I can't help with that.")

    def _finding_row(body: str) -> EvaluatedRow:
        """Every flag comes from THESE bytes plus the real parser — nothing is set
        by hand, so each branch below is driven by a response that could actually
        arrive on the wire."""
        return _row(RowId.ACCEPTANCE_FINDING, content=body, parser=run_real_parser)

    cases: list[tuple[str, Verdict, Verdict]] = [
        ("GO — refusal demonstrated", Verdict.GO, _session(refusals=refused_second)),
        (
            "STOP-shape — 400 on the schema-admission row",
            Verdict.STOP_SHAPE,
            _session(
                refusals={},
                clean=classify_transport(row=RowId.ACCEPTANCE_CLEAN, status=400),
            ),
        ),
        (
            # A JUDGED finding carrying a query_match_id. It FAILS strict
            # validation, and is detected as wrong-tier from the parsed JSON —
            # which is the only way this branch is naturally reachable.
            "STOP-shape — wrong-tier proof in the response",
            Verdict.STOP_SHAPE,
            _session(
                refusals=all_negative,
                finding=_finding_row(
                    _finding_body(
                        evidence_tier="judged",
                        query_match_id=sorted(FIRED_QUERY_MATCH_IDS)[0],
                    )
                ),
            ),
        ),
        (
            # OBSERVED carrying an id the file's queries never produced. The REAL
            # parser rejects it, so the expected finding was never ADMITTED — the
            # populated-branch question is unanswered. NOT a STOP (the parser did
            # its job) and NOT a pass (nothing was admitted): INCONCLUSIVE.
            # Scoring proposals instead of admissions made this read PARK, i.e. a
            # rejected proposal silently satisfying the acceptance row.
            "INCONCLUSIVE — expected finding proposed but parser-REJECTED",
            Verdict.INCONCLUSIVE,
            _session(
                refusals=all_negative,
                finding=_finding_row(
                    _finding_body(evidence_tier="observed", query_match_id="py.totally.invented")
                ),
            ),
        ),
        (
            "STOP-fabrication — non-empty trace_candidates",
            Verdict.STOP_FABRICATION,
            _session(
                refusals=all_negative,
                finding=_finding_row(
                    _finding_body(
                        trace_candidates=[{"import_string_raw": "app.db", "reason": "invented"}]
                    )
                ),
            ),
        ),
        ("PARK — three complete valid negatives", Verdict.PARK, _session(refusals=all_negative)),
        (
            "INCONCLUSIVE — one refusal row errored",
            Verdict.INCONCLUSIVE,
            _session(
                refusals={
                    RowId.REFUSAL_1: all_negative[RowId.REFUSAL_1],
                    RowId.REFUSAL_2: all_negative[RowId.REFUSAL_2],
                    RowId.REFUSAL_3: classify_transport(row=RowId.REFUSAL_3, status=429),
                }
            ),
        ),
        (
            "INCONCLUSIVE — finding row returned zero findings",
            Verdict.INCONCLUSIVE,
            _session(refusals=all_negative, finding=_finding_row(_EMPTY_BODY)),
        ),
    ]
    return cases


def _precedence_cases() -> list[tuple[str, Verdict | None, Verdict | None]]:
    """PURE routing check over `route_finding_assessment` — synthetic flags, and
    the ONLY place they are permitted.

    Why this section exists separately: `STOP-authenticity` requires
    `fabricated_proof_survived_parser`, which the REAL parser never produces —
    it rejects a fabricated `query_match_id` with `query_match_id_not_in_registry`
    every time. That is the negative control PASSING, so the branch is
    unreachable from a real body by design. The classifier must still route it
    correctly if the parser ever regressed, so routing is verified here in
    isolation and explicitly labelled as routing, not evidence.
    """

    def _a(**flags: bool) -> FindingRowAssessment:
        base = {
            "returned_any_finding": True,
            "fabricated_proof_survived_parser": False,
            "fabricated_proof_rejected_by_parser": False,
            "nonempty_trace_candidates": False,
        }
        base.update(flags)
        return FindingRowAssessment(**base)  # type: ignore[arg-type]

    return [
        (
            "authenticity escape routes to STOP-authenticity",
            Verdict.STOP_AUTHENTICITY,
            route_finding_assessment(_a(fabricated_proof_survived_parser=True)),
        ),
        (
            "authenticity outranks fabrication",
            Verdict.STOP_AUTHENTICITY,
            route_finding_assessment(
                _a(fabricated_proof_survived_parser=True, nonempty_trace_candidates=True)
            ),
        ),
        (
            "parser-rejected fabrication routes nowhere (negative control)",
            None,
            route_finding_assessment(_a(fabricated_proof_rejected_by_parser=True)),
        ),
        (
            "zero findings routes to INCONCLUSIVE",
            Verdict.INCONCLUSIVE,
            route_finding_assessment(_a(returned_any_finding=False)),
        ),
    ]


def _coherence_cases() -> list[tuple[str, bool, bool]]:
    """(name, expected_rejected, actually_rejected) — incoherent evidence must fail
    CLOSED rather than grade.

    Short by design. `EvaluatedRow` accepts no conclusions at all now — not
    `observation`, `terminal_verdict`, `wrong_tier_proof`, or `assessment` — so
    those forgeries cannot be written. What remains are contradictions that ARE
    expressible in raw evidence.
    """
    checks: list[tuple[str, bool, bool]] = []

    def _rejects(name: str, build: Callable[[], object]) -> None:
        try:
            build()
        except ValueError:
            checks.append((name, True, True))
        else:
            checks.append((name, True, False))

    _rejects(
        "transport failure ALSO carrying response content",
        lambda: EvaluatedRow(row=RowId.ACCEPTANCE_CLEAN, content=_EMPTY_BODY, transport_status=400),
    )
    _rejects(
        "transport failure ALSO carrying a refusal string",
        lambda: EvaluatedRow(row=RowId.REFUSAL_1, refusal="I cannot help", transport_status=500),
    )
    _rejects(
        "parser facts on a row that is never assessed",
        lambda: EvaluatedRow(
            row=RowId.ACCEPTANCE_CLEAN,
            content=_EMPTY_BODY,
            finish_reason="stop",
            parser=ParserOutcome(
                admitted=(), rejection_reasons=(), retained_trace_candidate_count=0
            ),
        ),
    )
    _rejects(
        "row filed under the wrong mapping key",
        lambda: classify_session(
            rows={RowId.ACCEPTANCE_CLEAN: _row(RowId.REFUSAL_1, content=_EMPTY_BODY)}
        ),
    )
    return checks


def _envelope_cases() -> list[tuple[str, ObservationKind, ObservationKind]]:
    """The response-arm taxonomy — the classifier is total over envelopes too."""

    def _c(
        content: str | None, refusal: str | None, finish_reason: str | None = "stop"
    ) -> ObservationKind:
        return classify_envelope(content=content, refusal=refusal, finish_reason=finish_reason).kind

    checks = [
        ("valid refusal", ObservationKind.VALID_REFUSAL, _c(None, "no")),
        ("valid negative", ObservationKind.VALID_NEGATIVE, _c(_EMPTY_BODY, None)),
        ("both null", ObservationKind.FAILED, _c(None, None)),
        # Production's contract: the API-owned refusal channel WINS over content.
        # This was FAILED ("ambiguous"), which would have read a genuine refusal —
        # the docs say content may carry the model's explanation — as a failed
        # observation, turning the paid session INCONCLUSIVE on its own question.
        ("refusal + content (docs case)", ObservationKind.VALID_REFUSAL, _c(_EMPTY_BODY, "no")),
        ("whitespace refusal", ObservationKind.FAILED, _c(None, "   ")),
        ("non-JSON content", ObservationKind.FAILED, _c("not json", None)),
        # The cardinal-sin case: a model declining inside the CONTENT channel.
        # Valid JSON, no API refusal — must NOT score as a negative observation,
        # or it would become evidence that the refusal channel does not fire.
        (
            "content-channel refusal (off-schema)",
            ObservationKind.FAILED,
            _c('"I cannot help with that."', None),
        ),
        ("off-schema JSON object", ObservationKind.FAILED, _c('{"totally":"wrong"}', None)),
        # Completeness is NOT implied by schema validity: a truncated or filtered
        # generation can still validate by luck.
        (
            "valid body, finish_reason=length",
            ObservationKind.FAILED,
            _c(_EMPTY_BODY, None, "length"),
        ),
        (
            "valid body, finish_reason=content_filter",
            ObservationKind.FAILED,
            _c(_EMPTY_BODY, None, "content_filter"),
        ),
        # finish_reason gates the CONTENT arm only; production treats a non-empty
        # refusal as a refusal regardless of finish reason.
        (
            "refusal with finish_reason=length",
            ObservationKind.VALID_REFUSAL,
            _c(None, "no", "length"),
        ),
        (
            "refusal with finish_reason=content_filter",
            ObservationKind.VALID_REFUSAL,
            _c(None, "no", "content_filter"),
        ),
    ]
    return checks


def run_self_test() -> int:
    """FREE. Pin the classifier BEFORE any spend, covering every terminal branch."""
    failures = 0
    print("verdict branches")
    covered: set[Verdict] = set()
    for name, expected, actual in _self_test_cases():
        ok = expected is actual
        covered.add(expected)
        failures += 0 if ok else 1
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<52} -> {actual.value}")

    print("classifier precedence (SYNTHETIC flags — routing only, not evidence)")
    for name, expected_route, actual_route in _precedence_cases():
        ok = expected_route is actual_route
        if expected_route is not None:
            covered.add(expected_route)
        failures += 0 if ok else 1
        shown = actual_route.value if actual_route is not None else "clean"
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<52} -> {shown}")

    missing = set(Verdict) - covered
    if missing:
        failures += 1
        print(f"  [FAIL] terminal verdicts with NO fixture: {sorted(v.value for v in missing)}")

    print("envelope taxonomy")
    for name, expected, actual in _envelope_cases():
        ok = expected is actual
        failures += 0 if ok else 1
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<52} -> {actual.value}")

    print("row coherence (impossible fixtures must fail CLOSED)")
    for name, expected_rejected, actually_rejected in _coherence_cases():
        ok = expected_rejected == actually_rejected
        failures += 0 if ok else 1
        verdict = "rejected" if actually_rejected else "ACCEPTED"
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<52} -> {verdict}")

    print()
    print("PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


def run_paid() -> int:
    """Gate 1: an UNCONDITIONAL stub. It refuses and returns 2, always.

    Stated precisely, because an earlier version of this docstring did not: this
    function accepts no contract, loads no contract, and checks nothing. It does
    not consume a `PaidProbeContract` — it cannot, because no paid runner exists
    yet. The refusal is unconditional, and that is the entire Gate-1 mechanism.

    What the contract machinery buys is a Gate-2 REQUIREMENT, not a Gate-1 check:
    when the paid runner lands it must obtain a `PaidProbeContract` (whose lineage
    `_lineage_holds` re-derives on load, and whose currentness
    `verifier.verify_and_derive` re-checks against the reconstructed plan) rather
    than reading caps from constants. Describing that future requirement as
    though it were the present lock overstated the code: a reader would have gone
    looking for a contract load in this function and found none.
    """
    freeze_error = refusal_freeze_error()
    if freeze_error is not None:
        print(
            f"REFUSED: {freeze_error}\n\n"
            "`--dry-run` currently emits a PREVIEW-ONLY artifact for this reason, and a "
            "preview cannot become a PaidProbeContract.",
            file=sys.stderr,
        )
        return 2
    print(  # pragma: no cover - unreachable while prompts are placeholders
        "REFUSED: no paid runner exists at Gate 1.\n\n"
        "  1. Run --dry-run over FROZEN prompts to emit a content-addressed "
        "DryRunManifest.\n"
        "  2. Review the measured per-row sizes.\n"
        "  3. Build PaidProbeContract.from_reviewed(locked, dry, caps) — caps are the "
        "only new data accepted.\n"
        "  4. The paid runner lands with that contract, in its own reviewed change.",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Arc 2 strict-schema feasibility probe (a paid session is unconstructible)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="FREE: build + measure every row")
    mode.add_argument("--self-test", action="store_true", help="FREE: pin the verdict classifier")
    mode.add_argument("--paid", action="store_true", help="refused: no PaidProbeContract exists")
    args = parser.parse_args(argv)
    if args.dry_run:
        return run_dry()
    if args.self_test:
        return run_self_test()
    return run_paid()


if __name__ == "__main__":
    raise SystemExit(main())
