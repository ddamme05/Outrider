"""The ordered attempt ledger — REFERENCES only. The fixture bytes are the evidence.

Three disjoint attempt kinds, discriminated on `kind`, each persisted whole as its
own fixture file:

- `CompletedCapture` — a response was normalized successfully. This includes a
  response whose CONTENT later fails strict validation: the content is retained,
  because wrong-tier proof must still be detectable from it. Strict validation is
  a verifier-derived judgment ABOUT this evidence, never a reason to discard it.
- `TransportFailure` — no completed response at all (HTTP error or timeout).
- `CaptureShapeFailure` — the wrapper could not project the response into a
  `RawCapture` (`RawCaptureShapeError`). A projection failure and an off-schema
  body are materially different; collapsing them would throw away a body that
  still carries evidence. It retains `sdk_response_json` for exactly that reason.

**One evidence authority.** The ledger holds ordinal + row + fixture digest and
NOTHING ELSE. An earlier version stored the response content, refusal, model, and
status on the ledger *beside* a fixture digest — so the verifier hashed one copy
and graded another, and an untouched fixture could accompany forged ledger
content. A `GO` was derivable from bytes that contained no refusal. There is now
exactly one place a response fact can come from: the hash-verified fixture,
decoded through the strict union.

**Structure only.** This module enforces what can be checked without reading a
body: order, prefix, contiguous ordinals, no retries, the five-call ceiling. The
SEMANTIC automaton — which attempts are reachable given what earlier responses
actually said — lives in the verifier, because it requires strict validation and
parsing that only the decoded evidence supports.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

# RawCapture is a runtime pydantic field annotation — it cannot move behind
# TYPE_CHECKING the way the two signature-only exception types can.
from outrider.llm.raw_openai_capture import RawCapture  # noqa: TC001
from spikes.openai.arc2.classifier import ROW_ORDER, RowId

if TYPE_CHECKING:
    from outrider.llm.raw_openai_capture import RawCaptureShapeError, RawOpenAICaptureError

__all__ = [
    "ATTEMPT_ADAPTER",
    "Attempt",
    "AttemptKind",
    "AttemptLedger",
    "AttemptRef",
    "CaptureShapeFailure",
    "CompletedCapture",
    "LedgerViolationError",
    "TransportFailure",
]

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
NonNegInt = Annotated[int, Field(ge=0)]
Bounded = Annotated[str, Field(min_length=1, max_length=500)]


class LedgerViolationError(Exception):
    """The attempt sequence is not one the frozen procedure can produce."""


class AttemptKind(StrEnum):
    COMPLETED = "completed"
    TRANSPORT_FAILURE = "transport_failure"
    CAPTURE_SHAPE_FAILURE = "capture_shape_failure"


class _Attempt(BaseModel):
    """Base for the persisted fixture payloads. Closed and frozen."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    ordinal: NonNegInt
    row: RowId
    #: The exact request-kwargs bytes this attempt sent. Checked against the
    #: contract's reviewed measurement during replay, so a response cannot be
    #: graded under a contract whose request it did not answer.
    request_body_digest: Sha256


class CompletedCapture(_Attempt):
    """A normalized completed response — wire facts only, no judgment.

    The boundary-owned `RawCapture` is nested WHOLE. Reads go through properties
    that delegate to it, so there is exactly one copy of every response fact —
    the same discipline the ledger follows, applied inside the fixture.
    `response_model` is the model the API SAID answered; replay refuses it when it
    disagrees with the contract.
    """

    kind: Literal[AttemptKind.COMPLETED] = AttemptKind.COMPLETED
    #: The WHOLE boundary-owned DTO, nested rather than flattened. An earlier
    #: version copied five fields out of it and dropped the rest, discarding
    #: `sdk_response_json` and `usage` — diagnostics the wrapper had already
    #: validated and which a failed paid run would need most. Nesting keeps one
    #: copy of every response fact instead of a lossy projection beside the source.
    capture: RawCapture

    @property
    def content(self) -> str | None:
        return self.capture.content

    @property
    def refusal(self) -> str | None:
        return self.capture.refusal

    @property
    def finish_reason(self) -> str | None:
        return self.capture.finish_reason

    @property
    def response_model(self) -> str:
        """Non-optional here by validation, unlike on `RawCapture`."""
        assert self.capture.response_model is not None  # noqa: S101 - pinned below
        return self.capture.response_model

    @property
    def response_id(self) -> str:
        assert self.capture.response_id is not None  # noqa: S101 - pinned below
        return self.capture.response_id

    @model_validator(mode="after")
    def _identity_is_present(self) -> Self:
        """A response that cannot say who answered it is not gradeable.

        `RawCapture` types `response_model`/`response_id` as `str | None` because
        the wrapper reports what the API sent, absence included. An attempt narrows
        them: replay compares the answering model against a model-bound contract, so
        a missing one leaves that comparison with nothing to compare. Enforced here
        rather than in `from_capture` so a fixture decoded straight off disk gets the
        same check as one built in-process.
        """
        if not self.capture.response_model or not self.capture.response_id:
            missing = "model" if not self.capture.response_model else "id"
            msg = (
                f"capture for {self.row.value!r} carries no {missing}: a response that "
                "cannot identify itself cannot be graded under a model-bound contract"
            )
            raise LedgerViolationError(msg)
        return self

    @classmethod
    def from_capture(
        cls, capture: RawCapture, *, ordinal: int, row: RowId, request_body_digest: str
    ) -> Self:
        """Wrap a validated capture as an attempt. No field is recomputed or defaulted.

        In particular a `None` refusal stays `None` — collapsing it to `""` would
        erase the very distinction the probe measures.
        """
        return cls(
            ordinal=ordinal, row=row, request_body_digest=request_body_digest, capture=capture
        )


class TransportFailure(_Attempt):
    """No completed response. `status` is None for a timeout.

    `request_id` and `message` are DIAGNOSTIC ONLY and no verdict branch reads
    them — classification is positional and status-derived, pinned by
    `test_classification_is_positional_not_prose`. They are retained because the
    wrapper already validated them (`_valid_request_id`, bounded excerpt) and a
    failed paid run is exactly when a vendor request id is worth having. Note the
    excerpt is deliberately unsanitized vendor text, fit only for the gitignored
    operator-local evidence rows — see `RawOpenAICaptureError`.
    """

    kind: Literal[AttemptKind.TRANSPORT_FAILURE] = AttemptKind.TRANSPORT_FAILURE
    status: Annotated[int, Field(ge=100, le=599)] | None
    request_id: Bounded | None = None
    message: Annotated[str, Field(max_length=500)] | None = None

    @classmethod
    def from_error(
        cls, error: RawOpenAICaptureError, *, ordinal: int, row: RowId, request_body_digest: str
    ) -> Self:
        """Retain the wrapper-validated transport diagnostics rather than dropping them."""
        return cls(
            ordinal=ordinal,
            row=row,
            request_body_digest=request_body_digest,
            status=error.status,
            request_id=error.request_id,
            message=error.message[:500] or None,
        )

    @model_validator(mode="after")
    def _not_a_success_status(self) -> Self:
        if self.status is not None and 200 <= self.status < 300:
            msg = (
                f"transport failure carries success status {self.status}: a 2xx is a "
                "completed response, not a transport failure"
            )
            raise ValueError(msg)
        return self


class CaptureShapeFailure(_Attempt):
    """The wrapper could not project the response — `RawCaptureShapeError` only.

    `sdk_response_json` is the whole point of keeping this variant distinct from a
    transport failure: a novel malformed shape is the most informative artifact the
    probe can produce, and dropping it (as an earlier version did) would leave a
    reason string describing evidence nobody kept. `None` only when the response
    was not serializable at all.
    """

    kind: Literal[AttemptKind.CAPTURE_SHAPE_FAILURE] = AttemptKind.CAPTURE_SHAPE_FAILURE
    reason: Bounded
    sdk_response_json: str | None = None

    @classmethod
    def from_error(
        cls, error: RawCaptureShapeError, *, ordinal: int, row: RowId, request_body_digest: str
    ) -> Self:
        return cls(
            ordinal=ordinal,
            row=row,
            request_body_digest=request_body_digest,
            reason=error.reason[:500],
            sdk_response_json=error.sdk_response_json,
        )


Attempt = Annotated[
    CompletedCapture | TransportFailure | CaptureShapeFailure,
    Field(discriminator="kind"),
]

#: Decodes a fixture's bytes into the closed union. The ONLY sanctioned path from
#: persisted evidence to a gradeable attempt.
ATTEMPT_ADAPTER: TypeAdapter[Attempt] = TypeAdapter(Attempt)


class AttemptRef(BaseModel):
    """A pointer to one attempt's fixture. Carries NO response facts.

    Deliberately minimal: anything stored here would be a second copy of a fact
    the fixture already holds, and two copies is how the forged-content attack
    worked.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    ordinal: NonNegInt
    row: RowId
    fixture_digest: Sha256


class AttemptLedger(BaseModel):
    """The ordered references, validated for STRUCTURE only.

    Reachability is not decidable here: whether a later attempt should exist
    depends on what earlier responses actually said, which requires the decoded
    evidence. That check lives in the verifier.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    refs: tuple[AttemptRef, ...]

    @model_validator(mode="after")
    def _validate_structure(self) -> Self:
        if not self.refs:
            msg = "empty ledger: a session with no attempts observed nothing"
            raise LedgerViolationError(msg)
        if len(self.refs) > len(ROW_ORDER):
            msg = (
                f"{len(self.refs)} attempts exceeds the frozen maximum {len(ROW_ORDER)}: "
                "the procedure is five calls, no retries"
            )
            raise LedgerViolationError(msg)
        for i, ref in enumerate(self.refs):
            if ref.ordinal != i:
                msg = f"ordinals are not contiguous from zero: position {i} has {ref.ordinal}"
                raise LedgerViolationError(msg)
        rows = tuple(r.row for r in self.refs)
        if len(set(rows)) != len(rows):
            msg = f"duplicate row in ledger {[r.value for r in rows]!r}: retries are forbidden"
            raise LedgerViolationError(msg)
        expected = ROW_ORDER[: len(rows)]
        if rows != expected:
            msg = (
                f"rows {[r.value for r in rows]!r} are not the frozen prefix "
                f"{[r.value for r in expected]!r}: execution order is part of the procedure"
            )
            raise LedgerViolationError(msg)
        return self

    @property
    def rows(self) -> tuple[RowId, ...]:
        return tuple(r.row for r in self.refs)
