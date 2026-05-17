# See specs/2026-05-16-audit-persister.md + DECISIONS.md#014/#016.
"""AuditPersister — durable single-class implementation of both Protocols.

Implements `LLMExchangePersister` (`llm/base.py`) AND `PhaseEventSink`
(`audit/sinks.py`) from one body, sharing transaction lifecycle and
session-per-call discipline.

Key invariants:

- **Append-only.** Only INSERT statements; PG trigger on `audit_events` blocks
  UPDATE/DELETE. The persister never issues mutating SQL outside of INSERT.
- **Idempotent on `event_id`.** PK conflicts treated as "already persisted";
  payload-equality verification on conflict raises
  `AuditPersisterIdempotencyConflict` on mismatch (loud-failure shape per
  the append-only invariant's spirit — a same-`event_id` re-emit with
  different content is a producer-side bug, not a silent discard).
- **Atomic LLMCallEvent + llm_call_content write** per `DECISIONS.md#016`.
  Both rows commit together via `session.begin()` or neither does.
- **Post-retention content-resurrection guard.** When the audit-row INSERT
  hits a conflict (existing audit row matches), the audit-conflict branch
  ALWAYS exits via `return` or `raise` — it NEVER falls through to a
  content INSERT (a check-then-INSERT pattern would race the retention
  sweep). The branch SELECTs the existing content row and compares it
  to the attempted write: if absent (retention purged), returns as
  metadata-only-replay state idempotent no-op; if present and matches,
  returns as idempotent no-op; if present and mismatches, raises
  `AuditPersisterIdempotencyConflict` with the content-field digest set.
  Content INSERT is reachable ONLY from the freshly-inserted-audit
  branch where no prior content can exist for this `event_id` (the
  audit-events PK would have caught it). Prevents resurrecting raw
  prompt/completion content that retention deliberately removed.
- **Metadata-only exception contract.** `AuditPersisterIdempotencyConflict`
  carries `event_id`, `mismatched_fields`, and `FieldDigest` (SHA-256 + length
  per field) — never raw `prompt`/`completion`/`payload`. Reason: logger
  formatting flows `str(exception)` into log records' `message`, which
  `RejectLLMContentFilter` (key-based) does not catch.
- **SQLAlchemy parameter-leak defense.** Raw SQLAlchemy exceptions
  (`IntegrityError`, `DataError`, etc.) include bound parameter values
  in their string representation by default — for a failing content
  INSERT, that surface would carry raw `prompt`/`completion` text. The
  persister relies on `hide_parameters=True` on the engine (set in
  `api/lifespan.py::_default_engine_factory`) to strip bound values
  from exception strings.

  Two complementary defenses converge here. The wrapper at
  `anthropic_provider.py::complete()` already type-narrows on
  `METADATA_ONLY_EXCEPTION_TYPES`: for any persister exception class
  listed in that tuple it renders `str(exc)` (each carries a
  contributor-enforced metadata-only `__str__`) and preserves the cause
  chain via `from exc`; for ANY OTHER exception (including raw
  SQLAlchemy errors that somehow surface), it renders only
  `<TypeName>` and uses `from None` to drop the cause chain entirely
  (`__suppress_context__` set, no traceback walk into the original).
  Allowlist completeness is structurally enforced by
  `test_every_persister_exception_is_metadata_only_listed`
  (`inspect`-based discovery; a new exception class added to this
  module without being added to the tuple fails the test).
  So even WITHOUT `hide_parameters=True`, a raw SQLAlchemy exception
  reaching the wrapper would render only its class name + suppressed
  cause. `hide_parameters=True` is defense-in-depth for the case where
  the SQLAlchemy exception's text was rendered elsewhere (a log
  formatter that called `str()` on the exception before the wrapper
  re-raised). Tests that construct their own engine outside the
  lifespan MUST honor the same setting to preserve the layered
  defense — the wrapper's type-narrow is the primary gate; the
  engine's `hide_parameters` is the secondary.

  (Round-30 codex audit fold: the prior docstring said content would
  leak via `LLMPersisterError(f"{exc!r}")` — the round-3-era wrap
  shape, before the round-9 + round-26 hardening. Current shape is
  documented above; verified against
  `src/outrider/llm/anthropic_provider.py::complete()` step 9
  `except Exception as exc` block.)

Design choices documented in the persister spec; reading the spec is the
faster path to context than re-deriving from this docstring.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, NamedTuple

# Runtime import: typed-kwarg signatures on the strict-keyword exception
# constructors (round-41 fold) reference UUID as a parameter annotation;
# TYPE_CHECKING-only would only suffice if annotations were string-quoted,
# but the runtime-validated parameter shape benefits from a real type at
# module import.
from uuid import UUID  # noqa: TC003 — runtime annotation needed for typed exception constructors

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

from outrider.db.models.audit_events import AuditEvent as AuditEventRow
from outrider.db.models.llm_call_content import LLMCallContent
from outrider.db.models.reviews import Review

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.audit.config import RetentionSettings
    from outrider.audit.events import LLMCallEvent, ReviewPhaseEvent
    from outrider.llm.base import LLMRequest, LLMResponse

__all__ = [
    "AuditPersister",
    "AuditPersisterConfigError",
    "AuditPersisterIdempotencyConflict",
    "AuditPersisterReviewIdMismatchError",
    "AuditPersisterReviewNotFoundError",
    "AuditPersisterSchemaInvariantError",
    "FieldDigest",
    "METADATA_ONLY_EXCEPTION_TYPES",
]


# Sentinel for "no top-level phase_key for this event type." `audit_events`
# has a top-level nullable `phase_key TEXT` column denormalized from the
# JSONB payload (genesis migration line 103, indexed by
# `ix_audit_events_review_phase_key`). The persister populates it from
# `event.phase_key` only for `ReviewPhaseEvent`; every other event type
# writes NULL there.
_NO_PHASE_KEY: Final[None] = None

# Pydantic's `model_dump(exclude=...)` expects an IncEx-compatible type;
# IncEx accepts `set[str]` but not `frozenset[str]`. Plain set; never mutated.
_EXCLUDE_FROM_PAYLOAD: Final[set[str]] = {"sequence_number"}


# The metadata-only contract per `DECISIONS.md#016` point 4 is a property
# of EACH exception type the persister raises — `str(exc)` and
# `repr(exc)` MUST contain only schema identifiers (event_id, table/column
# names, mismatched_fields, SHA-256 digests), never raw payload content.
# `LLMPersisterError` at `anthropic_provider.py::complete()` step 9
# (`except Exception as exc`) consults this tuple to type-narrow the
# wrap shape: known metadata-only types render via `f"...{exc}..."`
# with `from exc` (cause chain preserved — both wrapper message and
# cause are content-clean); unknown types render only as
# `f"...<{type(exc).__name__}>..."` with `from None`
# (`__suppress_context__=True` blocks traceback walking into the
# original exception's args/str). A new exception type added to this
# module MUST be reviewed for the metadata-only property AND added to
# this tuple — that's the explicit contributor contract.
METADATA_ONLY_EXCEPTION_TYPES: tuple[type[BaseException], ...] = ()
# Populated below after each class is defined; forward-ref-free.


class FieldDigest(NamedTuple):
    """Metadata-only digest of a mismatched field for conflict diagnostics.

    Carries SHA-256 + byte-length of each side. Never raw content. The
    exception that surfaces this namedtuple flows through `logger.exception()`
    into log records' `message` field, which `RejectLLMContentFilter` is
    key-based and does NOT pattern-match against (per FUP-023). Including
    only digests + lengths means a leaked exception cannot resurrect content.
    """

    existing_sha256: str
    attempted_sha256: str
    existing_length: int
    attempted_length: int


# ---------------------------------------------------------------------------
# Exceptions.
# ---------------------------------------------------------------------------


class _FrozenAllowlistMeta(type):
    """Metaclass that locks specific class attributes against reassignment.

    Round-44 sharp-edges fold: the round-43 `__init_subclass__` hardening
    blocks SUBCLASS declarations from overriding `_PARAM_HINTS` /
    `_INVARIANTS`, AND the literal-class-ref lookup blocks INSTANCE
    shadowing. But Python has no built-in "final class attribute" —
    any module imported after `persister.py` can do
    `AuditPersisterConfigError._PARAM_HINTS = MappingProxyType({"evil":
    user_prompt})` on the PARENT class itself. This metaclass closes
    that residual bypass by rejecting `setattr` on the locked attribute
    names. Pattern mirrors `enum.EnumMeta`'s member-write block.

    Locked names are declared on each class via `_FROZEN_ALLOWLIST_NAMES`
    (a `ClassVar[frozenset[str]]`). Reassignment of any locked name
    after class definition raises `AttributeError`.

    **Round-45 codex correction — bootstrap-the-freeze.** Each
    `_FROZEN_ALLOWLIST_NAMES` declaration MUST include the string
    `"_FROZEN_ALLOWLIST_NAMES"` itself. Otherwise an attacker can
    clear the freeze declaration first
    (`cls._FROZEN_ALLOWLIST_NAMES = frozenset()`), then reassign the
    now-unprotected allowlist. The self-referential inclusion makes
    the freeze declaration as inviolable as the allowlist it governs.
    The pin test `test_*_frozen_allowlist_names_is_itself_frozen`
    asserts this bootstrap on each adopting class.
    """

    def __setattr__(cls, name: str, value: object) -> None:
        # Class-definition-time `__init_subclass__` already validated the
        # subclass dict; this metaclass blocks POST-definition mutation
        # on either the parent OR a subclass.
        frozen = cls.__dict__.get("_FROZEN_ALLOWLIST_NAMES", frozenset())
        # Walk MRO for inherited frozen-name declarations too.
        for base in cls.__mro__:
            frozen = frozen | base.__dict__.get("_FROZEN_ALLOWLIST_NAMES", frozenset())
        if name in frozen:
            raise AttributeError(
                f"cannot reassign {cls.__name__}.{name} after class definition; "
                "this allowlist is class-level closed by the metadata-only contract "
                "(DECISIONS.md#016 + round-44 sharp-edges fold). Add a new entry via "
                "a deliberate PR editing the literal mapping in the class body."
            )
        super().__setattr__(name, value)


class AuditPersisterConfigError(ValueError, metaclass=_FrozenAllowlistMeta):
    """Eager construction-time validation failure on `AuditPersister.__init__`.

    Mirrors the `BuildGraphError` / `LLMMissingAPIKeyError` precedent: fail
    loud at construction, not on the first call.

    **Strict-keyword constructor (round-41 codex fold).** Takes only the
    typed `param_name` kwarg; the documentation hint is derived
    internally from a class-level allowlist. Cannot accept arbitrary
    positional strings — a future contributor writing
    `AuditPersisterConfigError(request.user_prompt)` gets a TypeError at
    construction.

    **Class-level allowlist (round-42 codex fold).** `param_name` must
    be one of the canonical AuditPersister constructor parameters; the
    constructor raises `ValueError` if not. Removes the round-41 `hint:
    str` kwarg (which still accepted arbitrary user-derived strings)
    and derives the hint internally. A future
    `AuditPersisterConfigError(param_name=user_input)` is rejected at
    construction unless `user_input` literally equals an allowlisted
    parameter name — and any new constructor parameter requires
    updating both the persister AND the allowlist here.

    **Subclass-override hardening (round-43 sharp-edges + adversarial
    fold).** Two structural defenses against the insider-threat
    subclass-escalation vector flagged by the round-43 audit:
    (a) `__init_subclass__` REJECTS any subclass that defines its own
    `_PARAM_HINTS` — preventing `class MyConfigError(AuditPersisterConfigError):
    _PARAM_HINTS = MappingProxyType({"evil": user_prompt})`; (b) the
    lookup at construction uses the LITERAL class reference
    `AuditPersisterConfigError._PARAM_HINTS` rather than the MRO-resolving
    `self._PARAM_HINTS`. Both layers are necessary: (a) catches the
    declaration; (b) catches any runtime injection via descriptor
    protocol or `setattr`. The rejection-message renders only the
    SHA-256 prefix of the rejected value (NOT the value itself) to
    close the round-43 sharp-edges H2 finding — even the gate's
    rejection message cannot leak the value that triggered it.
    """

    # Names declared here are class-attr-frozen by `_FrozenAllowlistMeta`:
    # `AuditPersisterConfigError._PARAM_HINTS = ...` (after class definition)
    # raises AttributeError. Round-44 fold closes the post-definition
    # reassignment bypass surfaced by the round-43 sharp-edges audit.
    #
    # **Round-45 codex correction — bootstrap-the-freeze gap.** The set
    # itself MUST appear in the set. Otherwise an attacker can clear the
    # freeze declaration first (`cls._FROZEN_ALLOWLIST_NAMES = frozenset()`),
    # then reassign the now-unprotected `_PARAM_HINTS`. Self-referential
    # inclusion makes the freeze declaration as inviolable as the
    # allowlist it governs.
    _FROZEN_ALLOWLIST_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"_PARAM_HINTS", "_FROZEN_ALLOWLIST_NAMES"}
    )

    _PARAM_HINTS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "session_factory": "pass an async_sessionmaker[AsyncSession]",
            "retention_settings": "pass a RetentionSettings instance",
        }
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Forbid subclasses from overriding the allowlist. A subclass that
        # widens `_PARAM_HINTS` could be picked up by `isinstance(...,
        # METADATA_ONLY_EXCEPTION_TYPES)` via MRO, then render a wider
        # set of values through `str(exc)`. Reject the declaration at
        # class-creation time so insider-threat / test-file subclass
        # escalation fails-loud at import.
        if "_PARAM_HINTS" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} cannot override AuditPersisterConfigError._PARAM_HINTS; "
                "the allowlist is class-level closed by the metadata-only contract "
                "(DECISIONS.md#016). Add a new parameter to the parent class allowlist "
                "via a deliberate PR, not via subclass override."
            )

    def __init__(self, *, param_name: str) -> None:
        # Use LITERAL class reference for the allowlist lookup — not
        # `self._PARAM_HINTS` (MRO-resolving) — so a subclass that
        # somehow attached a `_PARAM_HINTS` attribute at runtime (via
        # `setattr` or descriptor protocol, bypassing __init_subclass__)
        # still sees the parent's allowlist at validation time.
        if param_name not in AuditPersisterConfigError._PARAM_HINTS:
            # Render only the SHA-256 prefix of the rejected value, NOT
            # the value itself. The rejected value MAY be content-bearing
            # by definition (the gate exists to catch that case). The
            # 12-char hex prefix is correlation-stable for operator
            # debugging without leaking what was sent.
            digest = hashlib.sha256(param_name.encode("utf-8", errors="replace")).hexdigest()[:12]
            raise ValueError(
                f"param_name must be one of "
                f"{sorted(AuditPersisterConfigError._PARAM_HINTS)}; "
                f"got value with sha256-prefix={digest!r}, length={len(param_name)}. "
                f"To add a new parameter, update AuditPersister.__init__ AND "
                f"this allowlist together."
            )
        super().__init__(
            f"{param_name} must not be None; {AuditPersisterConfigError._PARAM_HINTS[param_name]}"
        )
        self.param_name = param_name


class AuditPersisterReviewNotFoundError(LookupError):
    """`persist()` could not resolve `event.review_id` to a reviews row.

    The persister sources `installation_id` from `reviews.installation_id`
    inside the transaction; absence means a producer-side bug, since the
    reviews row must be created before the graph dispatches. Surfacing
    loud here is preferable to silently writing a content row with a
    fabricated installation_id (which would then violate the
    `llm_call_content.installation_id` FK regardless).

    **Strict-keyword constructor (round-41 codex fold).** Takes only the
    typed `review_id: UUID`; generates the canonical message from it.
    Cannot accept arbitrary positional strings.
    """

    def __init__(self, *, review_id: UUID) -> None:
        super().__init__(
            f"persist() requires a reviews row for review_id={review_id}; "
            "reviews row must be created before graph dispatch."
        )
        self.review_id = review_id


class AuditPersisterReviewIdMismatchError(ValueError):
    """`persist()` was called with `event.review_id != request.review_id`.

    The persister sources `installation_id` from `reviews WHERE id =
    event.review_id` and stores `request.user_prompt` + `response.text`
    in `llm_call_content` keyed by that installation. If the request's
    `review_id` disagrees with the event's, the call would store
    Review A's prompt/completion under Review B's installation scope —
    misattributing the audit trail.

    Today's `AnthropicProvider.complete()` builds `LLMCallEvent` from
    the `LLMRequest` (so `event.review_id == request.review_id` always),
    but `LLMExchangePersister` is a public Protocol that future
    providers / test mocks could violate. This check is a metadata-only
    fail-loud guard at the persister boundary.

    **Strict-keyword constructor (round-41 codex fold).** Takes only the
    two typed UUIDs; generates the canonical message from them. Cannot
    accept arbitrary positional strings.

    Metadata-only by contract: the exception carries the two UUIDs +
    field names only; no payload content.
    """

    def __init__(self, *, event_review_id: UUID, request_review_id: UUID) -> None:
        super().__init__(
            f"persist() called with mismatched review_ids: "
            f"event.review_id={event_review_id} but "
            f"request.review_id={request_review_id}; "
            "producer must build LLMCallEvent.review_id from "
            "LLMRequest.review_id (or vice versa). Persister refuses "
            "to attribute content across review scopes."
        )
        self.event_review_id = event_review_id
        self.request_review_id = request_review_id


class AuditPersisterSchemaInvariantError(RuntimeError, metaclass=_FrozenAllowlistMeta):
    """Schema-level invariant violation detected at runtime.

    Raised when the persister observes a state the schema's NOT-NULL /
    FK / append-only contracts should make impossible (e.g., a SELECT
    on `audit_events.payload` after a PK conflict returns None, which
    the column's NOT NULL constraint forbids).

    **Strict-keyword constructor (round-41 codex fold).** Takes typed
    `event_id: UUID` + `invariant: str` (the invariant identifier, e.g.,
    `"audit_events.payload NOT NULL"`). Cannot accept arbitrary positional
    strings — same defense as the sibling classes.

    **Class-level invariant allowlist (round-42 codex fold).** `invariant`
    must be in the canonical allowlist `_INVARIANTS`; the constructor
    raises `ValueError` if not. A future
    `AuditPersisterSchemaInvariantError(event_id=..., invariant=f"bad
    value {user_input}")` is rejected at construction unless the string
    literally matches an allowlisted invariant identifier. Adding a new
    schema-invariant violation site requires updating the allowlist
    here, which forces review.

    **Metadata-only by contract** per `DECISIONS.md#016` point 4 — the
    exception message MUST carry only schema-level identifiers (event_id,
    table name, column name), never payload content. Listed in
    `METADATA_ONLY_EXCEPTION_TYPES` so the LLM-wrapper exception
    translation can render it via `str()` safely; future authors editing
    this class MUST preserve the metadata-only property.
    """

    # Names class-attr-frozen by `_FrozenAllowlistMeta` per the round-44
    # sharp-edges fold (see AuditPersisterConfigError for rationale).
    # Round-45 codex correction: self-referential inclusion of
    # `_FROZEN_ALLOWLIST_NAMES` itself closes the bootstrap-the-freeze
    # bypass (clear the freeze declaration first, then reassign the
    # now-unprotected `_INVARIANTS`).
    _FROZEN_ALLOWLIST_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"_INVARIANTS", "_FROZEN_ALLOWLIST_NAMES"}
    )

    _INVARIANTS: ClassVar[frozenset[str]] = frozenset(
        {
            "audit_events.payload NOT NULL",
        }
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Same subclass-override defense as AuditPersisterConfigError per
        # the round-43 fold — see that class for the rationale.
        if "_INVARIANTS" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} cannot override AuditPersisterSchemaInvariantError._INVARIANTS; "
                "the allowlist is class-level closed by the metadata-only contract "
                "(DECISIONS.md#016). Add a new invariant via a deliberate PR, "
                "not via subclass override."
            )

    def __init__(self, *, event_id: UUID, invariant: str) -> None:
        # Literal class reference for the allowlist lookup; see the
        # round-43 hardening rationale on AuditPersisterConfigError.
        if invariant not in AuditPersisterSchemaInvariantError._INVARIANTS:
            digest = hashlib.sha256(invariant.encode("utf-8", errors="replace")).hexdigest()[:12]
            raise ValueError(
                f"invariant must be one of "
                f"{sorted(AuditPersisterSchemaInvariantError._INVARIANTS)}; "
                f"got value with sha256-prefix={digest!r}, length={len(invariant)}. "
                f"Adding a new schema-invariant violation site requires updating "
                f"the allowlist on this class — per the round-42 metadata-only "
                f"contract enforcement."
            )
        super().__init__(f"{invariant} for event_id={event_id}; schema invariant violated")
        self.event_id = event_id
        self.invariant = invariant


class AuditPersisterIdempotencyConflict(ValueError):  # noqa: N818 — spec-defined name; "Conflict" is the semantic category, not "Error"
    """Same `event_id` re-emission with different content.

    Metadata-only by contract per `DECISIONS.md#016` point 4 (logs stay
    metadata-only — `RejectLLMContentFilter` is key-based and does not
    pattern-match exception text bound into log record `message` fields).
    Attributes:

    - `event_id`: the conflicting event id (UUID).
    - `mismatched_fields`: tuple of field names whose values differ between
      the existing row and the attempted write. **Authoritative** — this
      is the complete list of what differed; every mismatch surfaces here
      regardless of column type.
    - `field_digests`: SHA-256 + byte-length tuple (`FieldDigest`) for the
      subset of mismatched fields where digesting carries diagnostic
      signal. **Populated for content-bearing payload fields** (e.g., the
      `audit_events.payload` columns from `_compute_field_digests`, and the
      text fields `prompt` / `completion` from `_compute_content_field_digests`).
      **Intentionally NOT populated for small-primitive content-row columns**
      (`installation_id: int`, `is_eval: bool` — a digest of `True` vs
      `False` carries no information beyond the name in `mismatched_fields`).
      Pin test:
      `tests/unit/test_audit_persister.py::test_compute_content_field_digests_intentionally_omits_non_text_columns`.

    Consumers MUST treat `mismatched_fields` as the authoritative list and
    `field_digests` as best-effort detail for the fields where a digest is
    useful. `set(field_digests) ⊆ set(mismatched_fields)`; the reverse is
    not guaranteed.

    Operators investigating a conflict pull the full `audit_events.payload`
    and `llm_call_content` rows out-of-band (dashboard, `SELECT`); this
    exception is the SIGNAL, the DB is the SOURCE.
    """

    def __init__(
        self,
        *,
        event_id: UUID,
        mismatched_fields: tuple[str, ...],
        field_digests: Mapping[str, FieldDigest],
    ) -> None:
        # Enforce the docstring invariant `set(field_digests) ⊆ set(mismatched_fields)`
        # at construction time. A call site that swaps argument order, or
        # passes digests computed over a stale field set, would otherwise
        # ship a metadata-only-looking exception whose diagnostic claims
        # are internally inconsistent — `mismatched_fields` says one thing,
        # `field_digests` keys say another. Fail-loud on construction so
        # the bug surfaces at the offending site, not in operator forensics.
        # Metadata-only preserved: the assertion message names only field
        # names (which are class-level identifiers, never content).
        digest_keys = set(field_digests)
        mismatched_set = set(mismatched_fields)
        if not digest_keys.issubset(mismatched_set):
            extra = digest_keys - mismatched_set
            raise ValueError(
                f"field_digests keys must be subset of mismatched_fields; "
                f"got extra digest keys not in mismatched_fields: {sorted(extra)}"
            )
        self.event_id = event_id
        self.mismatched_fields = mismatched_fields
        self.field_digests = field_digests
        super().__init__(
            f"idempotency conflict on event_id={event_id} mismatched_fields={mismatched_fields}"
        )


# Populate the allowlist now that every named exception class above is
# defined. `LLMPersisterError` at `anthropic_provider.py` checks this
# tuple to decide the wrap shape: listed types render as `str(exc)`
# with `from exc` (cause chain preserved — both wrapper message and
# cause are content-clean); types NOT listed render only as
# `<TypeName>` with `from None` (cause chain suppressed via
# `__suppress_context__=True` so `traceback.format_exception` cannot
# walk into the original exception's args/str). Any new exception type
# added to this module MUST be reviewed for the metadata-only property
# AND added here — that's the explicit contributor contract documented
# above. Structurally enforced by
# `test_every_persister_exception_is_metadata_only_listed`.
METADATA_ONLY_EXCEPTION_TYPES = (
    AuditPersisterConfigError,
    AuditPersisterReviewNotFoundError,
    AuditPersisterReviewIdMismatchError,
    AuditPersisterSchemaInvariantError,
    AuditPersisterIdempotencyConflict,
)


# ---------------------------------------------------------------------------
# Helpers (private; tested via the persister's integration tests).
# ---------------------------------------------------------------------------


def _sha256_text(value: str) -> str:
    """SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json_value(value: Any) -> str:
    """SHA-256 hex digest of a JSON-encoded value (canonical separators).

    Encoding choice mirrors `audit/events.py::compute_finding_content_hash`:
    compact separators produce one canonical byte sequence per input.
    """
    serialized = json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)
    return _sha256_text(serialized)


def _json_value_length(value: Any) -> int:
    """Byte length of the canonical JSON encoding for length-diagnostics."""
    return len(
        json.dumps(value, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")
    )


# Sentinel for "key absent from this side of the comparison." Used to
# distinguish missing-key from present-with-None-value in `_diff_field_names`
# and `_compute_field_digests`. A naive `.get(k)` collapses both cases
# (returns None), so a future optional event field defaulting to None
# would silently pass payload-equality verification even when missing
# entirely. The sentinel forces "missing" to render as a distinct value
# both in the diff and in the digest output.
_MISSING: Final = object()


def _diff_field_names(
    existing: Mapping[str, Any],
    attempted: Mapping[str, Any],
) -> tuple[str, ...]:
    """Names of fields whose values differ between two JSON-payload dicts.

    Includes keys present in one side but not the other (treats missing
    as a mismatch — a payload-shape change between emissions is itself
    a producer bug worth surfacing). Uses the `_MISSING` sentinel rather
    than `.get(k)` so a key present with `None` on one side and absent
    on the other still surfaces as a mismatch.
    """
    keys = sorted(set(existing.keys()) | set(attempted.keys()))
    return tuple(k for k in keys if existing.get(k, _MISSING) != attempted.get(k, _MISSING))


def _compute_field_digests(
    existing: Mapping[str, Any],
    attempted: Mapping[str, Any],
) -> Mapping[str, FieldDigest]:
    """SHA-256 + length per field that differs. Missing-side fields render
    with a distinct SHA-256 (over a sentinel string) and length `-1`, so
    diagnostics communicate absence vs. zero-byte-value unambiguously."""
    digests: dict[str, FieldDigest] = {}
    for field_name in _diff_field_names(existing, attempted):
        existing_value = existing.get(field_name, _MISSING)
        attempted_value = attempted.get(field_name, _MISSING)
        digests[field_name] = FieldDigest(
            existing_sha256=_value_or_missing_sha256(existing_value),
            attempted_sha256=_value_or_missing_sha256(attempted_value),
            existing_length=_value_or_missing_length(existing_value),
            attempted_length=_value_or_missing_length(attempted_value),
        )
    return digests


# Distinct hash input for "missing field." `\x00` prefix can't appear in
# any valid JSON-encoded value (JSON forbids unescaped control chars), so
# this sentinel collisions-free vs. real payload content.
_MISSING_HASH_INPUT: Final[str] = "\x00<MISSING>"


def _value_or_missing_sha256(value: Any) -> str:
    """SHA-256 over the canonical JSON form, OR over a sentinel for missing."""
    if value is _MISSING:
        return _sha256_text(_MISSING_HASH_INPUT)
    return _sha256_json_value(value)


def _value_or_missing_length(value: Any) -> int:
    """JSON-encoded byte length, OR `-1` for missing (explicit absence marker)."""
    if value is _MISSING:
        return -1
    return _json_value_length(value)


def _diff_content_field_names(
    *,
    prompt_db: str,
    prompt_new: str,
    completion_db: str,
    completion_new: str,
    installation_id_db: int,
    installation_id_new: int,
    is_eval_db: bool,
    is_eval_new: bool,
) -> tuple[str, ...]:
    """Content-row mismatched-field names.

    Includes `prompt`/`completion` (text content) AND
    `installation_id`/`is_eval` (purge-scope + eval-isolation metadata).
    The latter two drive operational semantics: `installation_id`
    controls per-installation retention purge scope; `is_eval` controls
    dashboard filtering, sweep ignoring, and the eval-row integrity gate
    in `tests/eval/conftest.py`. A re-emission with same text but
    flipped `is_eval` would silently bury a production review's content
    under the eval flag — exactly the bug class the eval-isolation
    contract is designed to prevent.
    """
    mismatched: list[str] = []
    if prompt_db != prompt_new:
        mismatched.append("prompt")
    if completion_db != completion_new:
        mismatched.append("completion")
    if installation_id_db != installation_id_new:
        mismatched.append("installation_id")
    if is_eval_db != is_eval_new:
        mismatched.append("is_eval")
    return tuple(mismatched)


def _compute_content_field_digests(
    *,
    prompt_db: str,
    prompt_new: str,
    completion_db: str,
    completion_new: str,
) -> Mapping[str, FieldDigest]:
    """SHA-256 + byte-length per mismatched text field.

    `installation_id` and `is_eval` are intentionally OMITTED: they are
    small primitives (int, bool), not text content. The mismatched-field
    name is the diagnostic signal; an operator inspecting the conflict
    pulls the actual values from the DB. The digest map is reserved for
    content fields where the raw values would themselves be sensitive.
    """
    digests: dict[str, FieldDigest] = {}
    if prompt_db != prompt_new:
        digests["prompt"] = FieldDigest(
            existing_sha256=_sha256_text(prompt_db),
            attempted_sha256=_sha256_text(prompt_new),
            existing_length=len(prompt_db.encode("utf-8")),
            attempted_length=len(prompt_new.encode("utf-8")),
        )
    if completion_db != completion_new:
        digests["completion"] = FieldDigest(
            existing_sha256=_sha256_text(completion_db),
            attempted_sha256=_sha256_text(completion_new),
            existing_length=len(completion_db.encode("utf-8")),
            attempted_length=len(completion_new.encode("utf-8")),
        )
    return digests


def _serialize_event_payload(event: LLMCallEvent | ReviewPhaseEvent) -> dict[str, Any]:
    """Pydantic event → JSONB payload dict, JSON-normalized.

    Per `audit/events.py` module docstring: `mode="json"` (so UUIDs and
    datetimes stringify) and `exclude={"sequence_number"}` (DB-assigned at
    INSERT time; not part of the in-memory event identity).

    Additionally round-trips through `json.dumps`/`json.loads` so the
    in-memory dict matches what the DB will return after a JSONB
    round-trip. Defends against future event-subtype field types whose
    `model_dump(mode="json")` form keeps a Python-native type that JSONB
    deserialization would change (e.g., Decimal → float in some paths).
    Today's events are all JSON-safe types, so this normalization is a
    no-op; the defense is in place for V1.5+ schema extensions.

    Deliberately **no `default=` catchall**. `model_dump(mode="json")`
    is contractually required to return JSON-safe types only — every
    field on every concrete `AuditEventBase` subclass MUST be either a
    JSON-native type (str/int/float/bool/list/dict/None) or have a
    field-serializer that emits one. A `TypeError` from this `json.dumps`
    call means a producer added a field that violates that contract;
    failing loud at serialization time surfaces the producer bug at the
    persister boundary rather than silently coercing through `str(...)`.
    """
    raw = event.model_dump(mode="json", exclude=_EXCLUDE_FROM_PAYLOAD)
    normalized: dict[str, Any] = json.loads(json.dumps(raw))
    return normalized


# ---------------------------------------------------------------------------
# AuditPersister.
# ---------------------------------------------------------------------------


class AuditPersister:
    """Durable persister; implements `LLMExchangePersister` + `PhaseEventSink`.

    Constructor accepts dependencies via keyword args:

    - `session_factory: async_sessionmaker[AsyncSession]` — the per-call session
      factory. Each public method acquires a fresh `AsyncSession` from this
      factory (no session sharing across coroutines — `AsyncSession` is
      not concurrent-safe; V1.5 parallel-analyze fanout WILL issue concurrent
      `persist()` calls). Construct the factory with `expire_on_commit=False`
      so post-commit attribute access on returned ORM rows doesn't trigger
      a lazy refresh on a closed session.
    - `retention_settings: RetentionSettings` — operator-overridable TTL config.
      The persister reads `retention_settings.llm_content_retention_ttl` and
      writes `retention_expires_at = event.timestamp + ttl` explicitly per row.

    Both are required and validated eagerly at `__init__`. `None` raises
    `AuditPersisterConfigError` immediately. Mirrors the `build_graph` and
    `AnthropicProvider.__init__` precedents (fail-loud at construction; never
    fail at the first call).

    Deliberately NO `isinstance(session_factory, async_sessionmaker)` gate.
    Test factories that wrap or subclass `async_sessionmaker` for instrumentation
    must remain compatible; the type annotation is the static gate; runtime
    flexibility wins.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        retention_settings: RetentionSettings,
    ) -> None:
        if session_factory is None:
            raise AuditPersisterConfigError(param_name="session_factory")
        if retention_settings is None:
            raise AuditPersisterConfigError(param_name="retention_settings")
        self._session_factory = session_factory
        self._retention_settings = retention_settings

    # -- LLMExchangePersister surface ----------------------------------------

    async def persist(
        self,
        event: LLMCallEvent,
        request: LLMRequest,
        response: LLMResponse,
    ) -> None:
        """Persist LLMCallEvent + llm_call_content atomically per #016.

        Idempotent on `event.event_id`; payload-mismatch on PK conflict raises
        `AuditPersisterIdempotencyConflict` (metadata-only). Resurrection guard:
        if the audit row exists but the content row was purged by retention,
        returns as no-op rather than re-inserting content.

        Raises:
          AuditPersisterReviewIdMismatchError: `event.review_id` differs from
            `request.review_id` (producer-side bug; the persister would
            store request's content under event's installation scope, mis-
            attributing the audit trail).
          AuditPersisterReviewNotFoundError: `event.review_id` doesn't resolve
            in `reviews` (producer-side bug).
          AuditPersisterIdempotencyConflict: same-`event_id` re-emit with
            different content.
        """
        # Pre-tx consistency check: event and request MUST agree on review_id.
        # Today's AnthropicProvider.complete() builds the event from the
        # request (so they always agree), but LLMExchangePersister is a
        # public Protocol and future providers / test mocks could violate
        # this. Without the check, the persister would lookup installation_id
        # via event.review_id but store request's content (prompt/completion)
        # under that installation — misattributing audit trail.
        # Metadata-only failure: carries only the two UUIDs + field names.
        if request.review_id != event.review_id:
            raise AuditPersisterReviewIdMismatchError(
                event_review_id=event.review_id,
                request_review_id=request.review_id,
            )

        payload = _serialize_event_payload(event)
        retention_expires_at: datetime = (
            event.timestamp + self._retention_settings.llm_content_retention_ttl
        )

        async with self._session_factory() as session, session.begin():
            # Step 1: resolve installation_id from the reviews row. This row
            # is created upstream (webhook handler) before graph dispatch;
            # its absence is a producer-side bug.
            installation_id = await session.scalar(
                select(Review.installation_id).where(Review.id == event.review_id)
            )
            if installation_id is None:
                raise AuditPersisterReviewNotFoundError(review_id=event.review_id)

            # Step 2: INSERT audit_events with ON CONFLICT verification.
            audit_stmt = (
                postgresql_insert(AuditEventRow)
                .values(
                    event_id=event.event_id,
                    review_id=event.review_id,
                    event_type=event.event_type,
                    phase_key=_NO_PHASE_KEY,  # LLMCallEvent has no phase_key
                    timestamp=event.timestamp,
                    is_eval=event.is_eval,
                    payload=payload,
                )
                .on_conflict_do_nothing(index_elements=["event_id"])
                .returning(AuditEventRow.event_id)
            )
            inserted_audit = await session.scalar(audit_stmt)
            audit_row_already_existed = inserted_audit is None

            if audit_row_already_existed:
                # Audit-conflict branch: the audit row already existed when
                # we attempted INSERT. From this branch, we MUST NOT attempt
                # the content INSERT — doing so would race the retention
                # sweep (check-content-exists then INSERT is a fundamentally
                # racy pattern; sweep could delete the content row between
                # check and INSERT, and the INSERT would succeed because
                # there's no PK conflict, resurrecting raw content that
                # retention deliberately removed). Instead: verify the
                # audit payload matches, fetch existing content (if still
                # present) for the idempotency-match check, and return.
                # The content INSERT only happens from the freshly-inserted-
                # audit-row branch below.
                existing_payload = await session.scalar(
                    select(AuditEventRow.payload).where(AuditEventRow.event_id == event.event_id)
                )
                # The audit row's payload column is NOT NULL (per the schema);
                # ON CONFLICT firing guarantees the row exists. If existing_payload
                # came back None, the schema invariant is broken — fail loud rather
                # than silently substituting `{}` for diagnostics.
                if existing_payload is None:
                    raise AuditPersisterSchemaInvariantError(
                        event_id=event.event_id,
                        invariant="audit_events.payload NOT NULL",
                    )
                if existing_payload != payload:
                    raise AuditPersisterIdempotencyConflict(
                        event_id=event.event_id,
                        mismatched_fields=_diff_field_names(existing_payload, payload),
                        field_digests=_compute_field_digests(existing_payload, payload),
                    )

                # Audit payload matches. Now check content: if absent
                # (retention purged), respect the purge as an idempotent
                # no-op; if present, verify it matches our attempted write
                # for the idempotency contract; either way, RETURN — never
                # fall through to content INSERT from this branch.
                #
                # The SELECT includes `installation_id` and `is_eval`
                # alongside the text content. Those columns drive
                # operational semantics (purge scope + eval isolation);
                # comparing only text would let a re-emission with same
                # prompt/completion but different `installation_id` or
                # flipped `is_eval` pass silently. `retention_expires_at`
                # is intentionally EXCLUDED from the comparison — it
                # derives from `event.timestamp + retention_settings.ttl`,
                # so a TTL config change between deploys can legitimately
                # produce different values for the same event_id; that's
                # an operator-driven re-emission, not a producer bug.
                content_row = await session.execute(
                    select(
                        LLMCallContent.prompt,
                        LLMCallContent.completion,
                        LLMCallContent.installation_id,
                        LLMCallContent.is_eval,
                    ).where(LLMCallContent.event_id == event.event_id)
                )
                row_or_none = content_row.one_or_none()
                if row_or_none is None:
                    # Post-retention state: audit row exists (append-only),
                    # content row purged. Treat as metadata-only-replay
                    # idempotent no-op. Returning here is the resurrection
                    # guard — we never INSERT content for a previously-
                    # purged audit row.
                    return
                prompt_db, completion_db, installation_id_db, is_eval_db = row_or_none
                if (
                    prompt_db != request.user_prompt
                    or completion_db != response.text
                    or installation_id_db != installation_id
                    or is_eval_db != event.is_eval
                ):
                    raise AuditPersisterIdempotencyConflict(
                        event_id=event.event_id,
                        mismatched_fields=_diff_content_field_names(
                            prompt_db=prompt_db,
                            prompt_new=request.user_prompt,
                            completion_db=completion_db,
                            completion_new=response.text,
                            installation_id_db=installation_id_db,
                            installation_id_new=installation_id,
                            is_eval_db=is_eval_db,
                            is_eval_new=event.is_eval,
                        ),
                        field_digests=_compute_content_field_digests(
                            prompt_db=prompt_db,
                            prompt_new=request.user_prompt,
                            completion_db=completion_db,
                            completion_new=response.text,
                        ),
                    )
                return  # both audit and content match; idempotent no-op

            # Step 3 (freshly-inserted audit branch): INSERT llm_call_content.
            # Reachable ONLY when the audit row was newly inserted this
            # transaction (audit_row_already_existed == False). From this
            # branch, the content row cannot pre-exist for this event_id
            # (the audit-events PK constraint prevented a prior write under
            # the same event_id from succeeding without us seeing the
            # conflict above).
            #
            # Direct attribute access on `request.user_prompt` / `response.text`
            # bypasses the LLMRequest/LLMResponse field-serializer redaction
            # (which would persist "<redacted, N chars>" instead of the actual
            # prompt). Pydantic field validators ran at construction time;
            # attribute reads are raw.
            content_stmt = (
                postgresql_insert(LLMCallContent)
                .values(
                    event_id=event.event_id,
                    installation_id=installation_id,
                    prompt=request.user_prompt,
                    completion=response.text,
                    is_eval=event.is_eval,
                    retention_expires_at=retention_expires_at,
                )
                .on_conflict_do_nothing(index_elements=["event_id"])
                .returning(LLMCallContent.event_id)
            )
            inserted_content = await session.scalar(content_stmt)
            if inserted_content is None:
                # Conflict on content. Reachable only when a concurrent emit
                # landed both rows between our audit INSERT and this content
                # INSERT (a separate transaction inserted audit+content for
                # the same event_id, then committed, after our audit INSERT
                # succeeded but before ours could commit — extremely tight
                # race window since both INSERTs are inside our transaction).
                #
                # Use `.one_or_none()` so a subsequent retention purge that
                # ran between our INSERT and our SELECT returns as a no-op
                # rather than raising `NoResultFound`. Retention contract
                # wins over conflict detection.
                # Same SELECT + comparison shape as the audit-conflict
                # branch above: include installation_id + is_eval so a
                # concurrent emit with the same event_id but different
                # purge-scope / eval-flag cannot pass silently as idempotent.
                content_row = await session.execute(
                    select(
                        LLMCallContent.prompt,
                        LLMCallContent.completion,
                        LLMCallContent.installation_id,
                        LLMCallContent.is_eval,
                    ).where(LLMCallContent.event_id == event.event_id)
                )
                row_or_none = content_row.one_or_none()
                if row_or_none is None:
                    return  # purged between INSERT and SELECT; respect retention
                prompt_db, completion_db, installation_id_db, is_eval_db = row_or_none
                if (
                    prompt_db != request.user_prompt
                    or completion_db != response.text
                    or installation_id_db != installation_id
                    or is_eval_db != event.is_eval
                ):
                    raise AuditPersisterIdempotencyConflict(
                        event_id=event.event_id,
                        mismatched_fields=_diff_content_field_names(
                            prompt_db=prompt_db,
                            prompt_new=request.user_prompt,
                            completion_db=completion_db,
                            completion_new=response.text,
                            installation_id_db=installation_id_db,
                            installation_id_new=installation_id,
                            is_eval_db=is_eval_db,
                            is_eval_new=event.is_eval,
                        ),
                        field_digests=_compute_content_field_digests(
                            prompt_db=prompt_db,
                            prompt_new=request.user_prompt,
                            completion_db=completion_db,
                            completion_new=response.text,
                        ),
                    )

    # -- PhaseEventSink surface ----------------------------------------------

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        """Persist a ReviewPhaseEvent row to audit_events.

        Idempotent on `event.event_id`; payload-mismatch on PK conflict raises
        `AuditPersisterIdempotencyConflict`. No content side-table; no
        resurrection guard needed.

        Populates the top-level denormalized `phase_key` column from
        `event.phase_key` (typically `None` in V1; V1.5 parallel-analyze
        will populate per-file). V1.5's per-file index queries depend on
        this column being populated correctly today.
        """
        payload = _serialize_event_payload(event)

        async with self._session_factory() as session, session.begin():
            stmt = (
                postgresql_insert(AuditEventRow)
                .values(
                    event_id=event.event_id,
                    review_id=event.review_id,
                    event_type=event.event_type,
                    phase_key=event.phase_key,
                    timestamp=event.timestamp,
                    is_eval=event.is_eval,
                    payload=payload,
                )
                .on_conflict_do_nothing(index_elements=["event_id"])
                .returning(AuditEventRow.event_id)
            )
            inserted = await session.scalar(stmt)
            if inserted is None:
                existing_payload = await session.scalar(
                    select(AuditEventRow.payload).where(AuditEventRow.event_id == event.event_id)
                )
                if existing_payload is None:
                    raise AuditPersisterSchemaInvariantError(
                        event_id=event.event_id,
                        invariant="audit_events.payload NOT NULL",
                    )
                if existing_payload != payload:
                    raise AuditPersisterIdempotencyConflict(
                        event_id=event.event_id,
                        mismatched_fields=_diff_field_names(existing_payload, payload),
                        field_digests=_compute_field_digests(existing_payload, payload),
                    )
