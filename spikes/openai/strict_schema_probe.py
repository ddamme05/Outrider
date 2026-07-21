"""Arc 2 — strict-schema feasibility probe (the arc's ONLY entrypoint).

Answers two unverified questions with ONE bounded paid capture
(specs/2026-07-20-arc2-strict-schema-feasibility.md):

  1. Does the live API accept Outrider's exact proof-preserving,
     evidence-tier-discriminated strict `json_schema`?
  2. Does GPT-5.6 populate the API-owned `message.refusal` channel for it —
     the discriminator the production `json_object` shape does not have, and
     the reason the whole `openai` host is production-unadmitted
     (`DECISIONS.md#056`, 2026-07-19 amendment / FUP-246 trigger 2)?

**Gate 2 (current): the paid runner exists and SPENDS when authorized.**
`--dry-run` and `--self-test` remain free. `--paid` requires the literal
`--i-authorize-spend`; without it the run refuses before a client is constructed.

The paid procedure, in order: rebuild the reviewed `PaidProbeContract` from the
single dry-run manifest plus `plan.REVIEWED_CAPS`; preflight EVERY row against its
exact request digest and byte cap (any violation aborts at zero spend); reserve a
FRESH evidence directory, refusing rather than overwriting prior evidence; persist
the contract before call one; then call rows in frozen order inside ONE event loop,
persisting each fixture and rewriting `attempt_ledger.json` after every call, and
stopping on the `terminal_reason` the verifier reports. No retries. A transport or
shape failure is kept as evidence, not discarded. The verdict is derived from the
persisted ledger; nothing stored is authoritative.

**Caps and limits.** `plan.REVIEWED_CAPS` holds the caps a human reviewed on
2026-07-20 — the EXACT measured per-row sizes, no headroom, because each row's
request digest is already bound and re-derived, so any drift invalidates the
contract before a byte cap would matter. `max_completion_tokens` rides under the
profile-declared `token_limit_param`, which SHAPER v3 established is
`max_completion_tokens` for GPT-5.6. Caps are generation-affecting, so the contract
binds them into identity
— see `arc2/contracts.py`, and `arc2/verifier.py` for how a capture is replayed
against that identity.

Deliberately NOT invented up front: numeric caps before the real request bodies
exist would be conformance theater, and two implementations could spend
materially differently while both claiming to follow the spec.

Run (FREE):
    uv run --no-sync python spikes/openai/strict_schema_probe.py --dry-run
    uv run --no-sync python spikes/openai/strict_schema_probe.py --self-test

Run (SPENDS, up to five real calls):
    op run --env-file=.env -- uv run --no-sync python \\
      spikes/openai/strict_schema_probe.py --paid --i-authorize-spend
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
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

from outrider.llm.raw_openai_capture import (  # noqa: E402 - after sys.path bootstrap
    RawCaptureShapeError,
    RawOpenAICaptureClient,
    RawOpenAICaptureError,
)
from spikes.openai.arc2.attempts import (  # noqa: E402 - after sys.path bootstrap
    Attempt,
    AttemptLedger,
    AttemptRef,
    CaptureShapeFailure,
    CompletedCapture,
    TransportFailure,
)
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
    ContractViolationError,
    DryRunManifest,
    PaidProbeContract,
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
    REVIEWED_CAPS,
    Scenario,
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
from spikes.openai.arc2.verifier import (  # noqa: E402 - after sys.path bootstrap
    ReplayError,
    VerifiedSession,
    verify_and_derive,
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
    # The PUBLIC function, not the private `_measure`. Beyond the boundary point: this
    # used to iterate ROW_ORDER here while `registered_measurements()` iterated
    # `build_rows()` dict order, and replay compares the two tuples with order-sensitive
    # equality — so they agreed only because insertion order happened to match.
    measurements = registered_measurements()
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
        return FindingRowAssessment(**base)

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
    # Distinct names: `expected`/`actual` above are `Verdict`, these are
    # `ObservationKind`, and reusing the names makes them the same variable.
    for name, expected_kind, actual_kind in _envelope_cases():
        ok = expected_kind is actual_kind
        failures += 0 if ok else 1
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<52} -> {actual_kind.value}")

    print("row coherence (impossible fixtures must fail CLOSED)")
    for name, expected_rejected, actually_rejected in _coherence_cases():
        ok = expected_rejected == actually_rejected
        failures += 0 if ok else 1
        verdict = "rejected" if actually_rejected else "ACCEPTED"
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<52} -> {verdict}")

    print()
    print("PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


#: The exact literal the CLI must supply. A flag name, not a boolean, so the
#: authorization appears verbatim in the shell history that spent the money.
_SPEND_TOKEN: Final[str] = "--i-authorize-spend"


def _require_api_key() -> str:
    """Read the key at the LAST moment, and fail loudly if it is not a real key.

    Deliberately not module-level: importing this file must never require a key, or
    the free paths would stop being free.

    An unresolved `op://` reference is rejected explicitly. The repo runs commands
    under `op run`, so a forgotten wrapper leaves the literal secret-reference string
    in the environment — non-empty, so a truthiness check accepts it, and the run
    then burns its authorization on a guaranteed 401.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        msg = "OPENAI_API_KEY is empty or unset; a paid session cannot authenticate"
        raise RuntimeError(msg)
    if key.startswith("op://"):
        msg = (
            "OPENAI_API_KEY is an unresolved 1Password reference "
            f"({key[:12]}...): run under `op run --env-file=.env -- ...`"
        )
        raise RuntimeError(msg)
    return key


def _final_run_dir(out: Path, contract: PaidProbeContract) -> Path:
    return out / f"capture_{contract.digest[:16]}"


def _reserve_staging_dir(out: Path, contract: PaidProbeContract) -> Path:
    """Claim a STAGING directory, refusing if the final one already exists.

    Two failures shaped this. `exist_ok=True` let a rerun write into a directory
    that already held attempts, so a shorter second session left the first's later
    fixtures behind and a replay saw a prefix no session produced. Fixing that with
    a direct `exist_ok=False` on the final path then created a subtler one: setup
    could fail AFTER the directory existed (a contract that would not round-trip),
    leaving a permanently-poisoned name that made every healthy rerun collide.

    So setup happens in staging, and `capture_<digest>` comes into existence only
    once the contract has been validated — see `_promote`.
    """
    final = _final_run_dir(out, contract)
    if final.exists():
        msg = (
            f"evidence directory already exists: {final}\n"
            "A paid session never overwrites or merges into prior evidence. Move or "
            "remove it deliberately, then re-run."
        )
        raise RuntimeError(msg)
    staging = out / f".setup_{contract.digest[:16]}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def _promote(staging: Path, out: Path, contract: PaidProbeContract) -> Path:
    """Rename staging to the final evidence directory. Called only after validation."""
    final = _final_run_dir(out, contract)
    os.rename(staging, final)
    return final


def _discard_staging(staging: Path) -> None:
    """Remove a staging directory after a setup failure.

    Safe by construction, and worth stating why deletion is acceptable here at all
    given this is a paid-evidence path: staging is created microseconds earlier, no
    client exists yet, and the only file it can hold is the `paid_contract.json`
    this process just wrote. No capture can be lost because none has happened.
    """
    shutil.rmtree(staging, ignore_errors=True)


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write via a temp file + `os.replace`, so a reader never sees a partial file.

    The ledger is rewritten after every call; a crash mid-write would otherwise
    leave the one artifact replay depends on truncated.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _persist_contract(run_dir: Path, contract: PaidProbeContract) -> PaidProbeContract:
    """Write the contract, READ IT BACK, and return the decoded instance to grade.

    The ledger already worked this way; the contract did not — it was written and
    then the in-memory object was used, with a test asserting only that the file
    existed. That is the same "grade one copy, preserve another" shape the ledger
    fix removed, one sibling over: a serialization or write regression could report
    `GO` while leaving a durable artifact nobody can replay.

    Returning the DECODED contract is what closes it. The session is then graded
    under exactly the bytes a later audit will load, and a contract that cannot
    round-trip fails here — before a client exists.
    """
    path = run_dir / "paid_contract.json"
    _write_atomic(path, contract.model_dump_json().encode("utf-8"))
    decoded = PaidProbeContract.model_validate_json(path.read_bytes())
    if decoded != contract:
        msg = (
            "persisted paid contract does not round-trip: the durable artifact "
            "differs from the reviewed contract this session would grade under"
        )
        raise RuntimeError(msg)
    return decoded


def _persist_ledger(run_dir: Path, refs: list[AttemptRef]) -> AttemptLedger:
    """Persist the ledger after EVERY attempt and return what was written back.

    The in-memory ledger is not the artifact — after the process exits, a replay has
    only what is on disk. Writing it each call also means an aborted or crashed
    session still leaves a replayable prefix rather than fixtures no ledger names.
    Returned by RE-READING the file, so the session grades the same bytes a later
    audit will.
    """
    path = run_dir / "attempt_ledger.json"
    _write_atomic(path, AttemptLedger(refs=tuple(refs)).model_dump_json().encode("utf-8"))
    return AttemptLedger.model_validate_json(path.read_bytes())


async def _execute_one(
    client: RawOpenAICaptureClient,
    ordinal: int,
    row: RowId,
    kwargs: dict[str, Any],
    request_body_digest: str,
) -> Attempt:
    """One call, projected into the closed attempt union.

    Never raises for a wire outcome — a transport error and a shape error are both
    EVIDENCE, and discarding either loses the only artifact a failed paid run makes.
    """
    try:
        capture = await client.capture(**kwargs)
    except RawOpenAICaptureError as err:
        return TransportFailure.from_error(
            err, ordinal=ordinal, row=row, request_body_digest=request_body_digest
        )
    except RawCaptureShapeError as err:
        return CaptureShapeFailure.from_error(
            err, ordinal=ordinal, row=row, request_body_digest=request_body_digest
        )
    return CompletedCapture.from_capture(
        capture, ordinal=ordinal, row=row, request_body_digest=request_body_digest
    )


def _persist(run_dir: Path, attempt: Attempt) -> AttemptRef:
    """Write the attempt as its own fixture, referenced BY HASH OF THE BYTES ON DISK.

    Re-read after writing so the digest describes what a later replay will actually
    load, not what this process meant to write.
    """
    path = run_dir / f"{attempt.ordinal:02d}_{attempt.row.value}.json"
    _write_atomic(path, attempt.model_dump_json().encode("utf-8"))
    return AttemptRef(
        ordinal=attempt.ordinal,
        row=attempt.row,
        fixture_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def reviewed_paid_contract(manifest_path: Path) -> PaidProbeContract:
    """Rebuild the reviewed contract from the dry-run artifact plus the source caps.

    The caps live in source (`plan.REVIEWED_CAPS`) so they are reviewed in a diff;
    the manifest lives on disk so it is content-addressed. `from_reviewed` re-derives
    the transition, and `_lineage_holds` re-derives it again on load — neither is
    trusted alone.
    """
    raw = manifest_path.read_bytes()
    dry = DryRunManifest.model_validate_json(raw)
    return PaidProbeContract.from_reviewed(
        locked=registered_locked_contract(), dry=dry, caps=dict(REVIEWED_CAPS)
    )


def _preflight(contract: PaidProbeContract, rows: dict[RowId, dict[str, Any]]) -> list[str]:
    """Check EVERY row before call one. Returns violations; empty means go.

    Two independent statements of the same reviewed fact, both required:
    the exact `request_body_digest`, and the byte cap. The digest is the strong
    check and would catch any drift on its own; the cap is a second, cruder ceiling
    that holds even if the digest comparison were somehow wrong.

    Deliberately total BEFORE any call — a per-row check interleaved with sending
    would spend on rows 1..k-1 before discovering row k is out of contract, and the
    whole point of a five-call budget is that it is refused whole or spent whole.
    """
    violations: list[str] = []
    reviewed = {m.row_id: m for m in contract.measurements}
    for row in contract.row_order:
        if row not in rows:
            violations.append(f"{row.value}: no request was built")
            continue
        body = serialize_kwargs(rows[row]).encode("utf-8")
        digest = hashlib.sha256(body).hexdigest()
        m = reviewed.get(row)
        if m is None:
            violations.append(f"{row.value}: no reviewed measurement")
            continue
        if digest != m.request_body_digest:
            violations.append(
                f"{row.value}: request body digest {digest[:16]} != reviewed "
                f"{m.request_body_digest[:16]} — the prompt drifted since review"
            )
        cap = contract.per_row_prompt_byte_cap.get(row)
        if cap is None:
            violations.append(f"{row.value}: no reviewed cap")
        elif len(body) > cap:
            violations.append(f"{row.value}: {len(body):,} B exceeds reviewed cap {cap:,} B")
    return violations


async def _run_session(
    contract: PaidProbeContract,
    rows: dict[RowId, dict[str, Any]],
    run_dir: Path,
    api_key: str,
) -> VerifiedSession | None:
    """The whole session in ONE event loop, with the client closed in `finally`.

    Previously each call ran under its own `asyncio.run(...)`, so the second call
    reused an `AsyncOpenAI` whose connection pool was bound to a loop that had
    already closed — and nothing ever closed the client. One loop for the session,
    one close, guaranteed.
    """
    client = RawOpenAICaptureClient(
        api_key=api_key,
        base_url=OPENAI_PROFILE.base_url,
        # Explicit, not inherited: "no retries" is part of the five-call procedure,
        # and a wrapper default could change without this file failing a test.
        max_retries=0,
    )
    refs: list[AttemptRef] = []
    session: VerifiedSession | None = None
    try:
        for ordinal, row in enumerate(contract.row_order):
            digest = next(m.request_body_digest for m in contract.measurements if m.row_id == row)
            attempt = await _execute_one(client, ordinal, row, rows[row], digest)
            refs.append(_persist(run_dir, attempt))
            ledger = _persist_ledger(run_dir, refs)
            print(f"  [{ordinal}] {row.value}: {attempt.kind.value}")

            # Replay the PERSISTED ledger, not the in-memory list — the artifact is
            # the thing a later audit reads, so it is the thing the session grades.
            session = verify_and_derive(
                contract=contract,
                evaluation=registered_evaluation_contract(),
                scenario_source=DEFECT_SCENARIO.source,
                ledger=ledger,
                fixture_dir=run_dir,
            )
            if session.terminal_reason is not None:
                print(f"  stopping: {session.terminal_reason}")
                break
    finally:
        await client.close()
    return session


def run_paid(*, confirm: str | None = None, out_dir: Path | None = None) -> int:
    """SPENDS MONEY. Up to five calls, one model, one session, no retries.

    Refuses unless `confirm` is exactly `_SPEND_TOKEN`. A reviewed contract states
    the experiment is well-formed; it does not state that now is the moment to buy
    it, so authorization is a separate act.

    Order, none of it skippable:

    1. Authorization token, then frozen prompts.
    2. Rebuild the reviewed contract from the dry-run artifact + source caps.
    3. Preflight EVERY row (exact request digest + byte cap). Any violation aborts
       at zero spend.
    4. Reserve a FRESH evidence directory and persist the contract before call one,
       so the artifact set is self-describing even if the session dies mid-run.
    5. Call rows in frozen order in ONE event loop; persist each fixture and rewrite
       the ledger after every call; replay the persisted ledger and stop on the
       `terminal_reason` the verifier reports.
    6. Print the derived verdict. Nothing is stored as authoritative.
    """
    if confirm != _SPEND_TOKEN:
        print(
            "REFUSED: paid execution requires explicit authorization.\n\n"
            f"  Re-run with: --paid {_SPEND_TOKEN}\n"
            f"  This sends up to {len(ROW_ORDER)} real requests to {MODEL}.",
            file=sys.stderr,
        )
        return 2

    freeze_error = refusal_freeze_error()
    if freeze_error is not None:
        print(f"REFUSED: {freeze_error}", file=sys.stderr)
        return 2

    out = out_dir if out_dir is not None else _OUT_DIR
    manifests = sorted(out.glob("dry_run_[0-9a-f]*.json"))
    if len(manifests) != 1:
        print(
            f"REFUSED: expected exactly one reviewed dry-run manifest in {out}, found "
            f"{len(manifests)}. Which manifest was reviewed is not resolvable by "
            "picking one.",
            file=sys.stderr,
        )
        return 2

    try:
        contract = reviewed_paid_contract(manifests[0])
    except (ValueError, ContractViolationError) as exc:
        print(f"REFUSED: reviewed contract does not rebuild: {exc}", file=sys.stderr)
        return 2

    rows = build_rows()
    violations = _preflight(contract, rows)
    if violations:
        print("REFUSED at preflight — ZERO spend. Violations:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 2

    # Credentials BEFORE the directory is claimed. Validating inside the session
    # meant a missing or unresolved key spent nothing but still left a poisoned
    # `capture_<digest>` behind — holding no replayable ledger, and (because the
    # name is deterministic) causing the correctly-wrapped rerun to refuse as a
    # collision. Failing here leaves the filesystem exactly as it was found.
    try:
        api_key = _require_api_key()
    except RuntimeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    try:
        staging = _reserve_staging_dir(out, contract)
    except (RuntimeError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    # The contract lands BEFORE call one, and is read back: evidence that cannot
    # say what it was captured under is not replayable, and a session can die at
    # any point. The DECODED instance is what the session grades under. A failure
    # here discards staging, so the final name stays unclaimed and a healthy rerun
    # does not collide with a failed setup.
    try:
        contract = _persist_contract(staging, contract)
        run_dir = _promote(staging, out, contract)
    except (RuntimeError, ValueError, OSError) as exc:
        _discard_staging(staging)
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"preflight OK: {len(rows)} rows within reviewed digests + caps")
    print(f"contract      : {contract.digest[:16]}")
    print(f"model         : {contract.model}")
    print(f"evidence      : {run_dir}")
    print(f"SPENDING NOW  : up to {len(contract.row_order)} calls\n")

    # `verify_and_derive` runs INSIDE the session (after each call), so its failures
    # surface here: a responding model that disagrees with the contract, a fixture
    # whose bytes no longer match its reference, a row filed under the wrong key.
    # Neither `ReplayError` nor `ContractViolationError` inherits `RuntimeError`, so
    # catching only that let them escape `asyncio.run` as an uncaught traceback with
    # exit 1 — after spend had already begun, which is exactly when a legible abort
    # matters. `OSError` is included because the fixture and ledger writes are disk
    # I/O on the same path.
    try:
        session = asyncio.run(_run_session(contract, rows, run_dir, api_key))
    except (RuntimeError, ReplayError, ContractViolationError, OSError) as exc:
        print(f"ABORTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"evidence retained: {run_dir}", file=sys.stderr)
        return 2

    if session is None:  # pragma: no cover - row_order is non-empty
        print("REFUSED: no attempt was made", file=sys.stderr)
        return 2

    print(f"\nverdict   : {session.verdict.verdict.value}")
    print(f"reason    : {session.verdict.reason}")
    print(f"calls     : {session.attempts_replayed}")
    print(f"evidence  : {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Arc 2 strict-schema feasibility probe. --dry-run/--self-test are FREE; "
            "--paid SPENDS and requires the literal authorization flag."
        ),
        # Abbreviation OFF: argparse would accept `--i` for the spend token, which
        # defeats the reason the flag is a literal — that the authorization appears
        # verbatim in the shell history of whoever spent the money.
        allow_abbrev=False,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="FREE: build + measure every row")
    mode.add_argument("--self-test", action="store_true", help="FREE: pin the verdict classifier")
    mode.add_argument("--paid", action="store_true", help="SPENDS: up to 5 real calls")
    parser.add_argument(
        _SPEND_TOKEN,
        dest="authorize",
        action="store_true",
        help="required with --paid; the literal token that authorizes real spend",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        return run_dry()
    if args.self_test:
        return run_self_test()
    return run_paid(confirm=_SPEND_TOKEN if args.authorize else None)


if __name__ == "__main__":
    raise SystemExit(main())
