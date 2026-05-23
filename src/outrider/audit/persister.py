# See specs/2026-05-16-audit-persister.md + DECISIONS.md#014/#016.
"""AuditPersister — durable single-class implementation of five sink Protocols.

Implements `LLMExchangePersister` (`llm/base.py`) AND `PhaseEventSink`,
`FileExaminationSink`, `AnalyzeEventSink`, and `PublishEventSink`
(`audit/sinks.py`) from one body, sharing transaction lifecycle and
session-per-call discipline. The non-phase events route through a
shared `_persist_non_phase_event` helper so the idempotency +
payload-mismatch discipline is uniform across event types. The
PublishEventSink surface was added per DECISIONS.md #023 (publish
routing and eligibility are separate decisions).

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

  (Earlier docstring referenced a `LLMPersisterError(f"{exc!r}")` wrap
  shape that has since been replaced by the two-layer defense described
  above; verified against `src/outrider/llm/anthropic_provider.py::complete()`
  step 9 `except Exception as exc` block.)

Design choices documented in the persister spec; reading the spec is the
faster path to context than re-deriving from this docstring.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, NamedTuple

# Runtime import: typed-kwarg signatures on the strict-keyword exception
# constructors reference UUID as a parameter annotation. TYPE_CHECKING-only
# would only suffice if annotations were string-quoted; runtime-validated
# parameter shapes benefit from a real type at module import.
from uuid import UUID  # noqa: TC003 — runtime annotation needed for typed exception constructors

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

from outrider.db.models.audit_events import AuditEvent as AuditEventRow
from outrider.db.models.llm_call_content import LLMCallContent
from outrider.db.models.reviews import Review
from outrider.llm.base import _canonical_prompt_hash, _canonical_system_prompt_hash
from outrider.llm.pricing import PRICING_VERSION, compute_cost_usd

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.audit.config import RetentionSettings
    from outrider.audit.events import (
        AgentTransitionEvent,
        AnalyzeCompletedEvent,
        AnalyzeResponseRejectedEvent,
        FileExaminationEvent,
        FindingEvent,
        FindingProposalRejectedEvent,
        LLMCallEvent,
        PublishAttemptEvent,
        PublishEligibilityEvent,
        PublishRoutingEvent,
        ReviewPhaseEvent,
    )
    from outrider.llm.base import LLMRequest, LLMResponse

# PublishEvent is consumed at RUNTIME by `query_prior_publish_event`'s
# `PublishEvent.model_validate(payload)` call — must be imported at
# module level, not under TYPE_CHECKING.
from outrider.audit.events import (
    PublishEvent,  # noqa: E402  (intentional post-TYPE_CHECKING runtime import)
)

__all__ = [
    "AuditPersister",
    "AuditPersisterConfigError",
    "AuditPersisterEventRequestFieldMismatchError",
    "AuditPersisterEventResponseFieldMismatchError",
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
    """Metaclass that rejects `setattr` on class attribute names listed in
    `_FROZEN_ALLOWLIST_NAMES` (a `ClassVar[frozenset[str]]` declared on
    the adopting class). Closes the "Python has no `final` class
    attribute" gap. Pattern mirrors `enum.EnumMeta`'s member-write block.

    `_FROZEN_ALLOWLIST_NAMES` MUST include its own name — otherwise an
    attacker can clear the freeze declaration first, then reassign the
    now-unprotected target. The self-referential inclusion is the
    bootstrap. See pin tests `test_*_frozen_allowlist_names_is_itself_frozen`.
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
                "(DECISIONS.md#016). Add a new entry via a deliberate PR editing "
                "the literal mapping in the class body."
            )
        super().__setattr__(name, value)


class AuditPersisterConfigError(ValueError, metaclass=_FrozenAllowlistMeta):
    """Eager construction-time validation failure on `AuditPersister.__init__`.

    Mirrors `BuildGraphError` / `LLMMissingAPIKeyError`: fail loud at
    construction, not on first call.

    Metadata-only contract: the strict-keyword `param_name: str` kwarg
    is the only construction surface; the hint is derived internally
    from `_PARAM_HINTS` so callers cannot inject arbitrary strings.
    `param_name` must match an allowlisted value or construction raises
    `ValueError`. `__init_subclass__` blocks subclass override of the
    allowlist, the metaclass blocks parent-class reassignment, and the
    rejection message renders only `sha256-prefix={hex[:12]}` so the
    gate itself doesn't leak the rejected value.
    """

    # `_PARAM_HINTS` is frozen against post-definition reassignment by
    # `_FrozenAllowlistMeta`. Self-reference closes the bootstrap bypass.
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
        # Subclass override of `_PARAM_HINTS` would widen the allowlist
        # while still matching `isinstance(..., METADATA_ONLY_EXCEPTION_TYPES)`
        # via MRO. Reject at class-creation time.
        if "_PARAM_HINTS" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} cannot override AuditPersisterConfigError._PARAM_HINTS; "
                "the allowlist is class-level closed by the metadata-only contract "
                "(DECISIONS.md#016)."
            )

    def __init__(self, *, param_name: str) -> None:
        # Literal class reference (not `self._PARAM_HINTS`) so a runtime
        # attribute injection bypassing `__init_subclass__` still hits
        # the parent's allowlist.
        if param_name not in AuditPersisterConfigError._PARAM_HINTS:
            # SHA-256 prefix instead of the raw value — the rejected
            # value may itself be content-bearing (the gate exists to
            # catch that case).
            digest = hashlib.sha256(param_name.encode("utf-8", errors="replace")).hexdigest()[:12]
            raise ValueError(
                f"param_name must be one of "
                f"{sorted(AuditPersisterConfigError._PARAM_HINTS)}; "
                f"got value with sha256-prefix={digest!r}, length={len(param_name)}."
            )
        super().__init__(
            f"{param_name} must not be None; {AuditPersisterConfigError._PARAM_HINTS[param_name]}"
        )
        self.param_name = param_name


class AuditPersisterReviewNotFoundError(LookupError):
    """`persist()` could not resolve `event.review_id` to a reviews row.

    Absence is a producer-side bug: the reviews row must exist before
    graph dispatch. Surfacing loud here beats silently writing a
    content row with a fabricated installation_id (which would then
    violate the `llm_call_content.installation_id` FK regardless).

    Strict-keyword `review_id: UUID` constructor; the message is
    generated from the UUID so no caller can inject content.
    """

    def __init__(self, *, review_id: UUID) -> None:
        super().__init__(
            f"persist() requires a reviews row for review_id={review_id}; "
            "reviews row must be created before graph dispatch."
        )
        self.review_id = review_id


class AuditPersisterReviewIdMismatchError(ValueError):
    """`persist()` was called with `event.review_id != request.review_id`.

    Without this guard, the call would store Review A's prompt/completion
    under Review B's installation scope (the persister keys
    `llm_call_content` by `installation_id` sourced from
    `reviews WHERE id = event.review_id`). `AnthropicProvider.complete()`
    builds `LLMCallEvent` from `LLMRequest` so the two always agree
    today, but `LLMExchangePersister` is a public Protocol that a
    future provider or test mock could violate.

    Strict-keyword `event_review_id: UUID` + `request_review_id: UUID`
    constructor; message generated from the two UUIDs.
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

    Strict-keyword `event_id: UUID` + `invariant: str` constructor;
    `invariant` must match an allowlisted identifier in `_INVARIANTS`
    or construction raises `ValueError`. `__init_subclass__` blocks
    subclass override of the allowlist; the metaclass blocks
    parent-class reassignment; the rejection message renders only
    `sha256-prefix={hex[:12]}`. Listed in
    `METADATA_ONLY_EXCEPTION_TYPES` per `DECISIONS.md#016` point 4
    — `str(exc)` carries only schema identifiers, never payload content.
    """

    # `_INVARIANTS` is frozen against post-definition reassignment by
    # `_FrozenAllowlistMeta`. Self-reference closes the bootstrap bypass.
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
        # See AuditPersisterConfigError for the subclass-override rationale.
        if "_INVARIANTS" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} cannot override AuditPersisterSchemaInvariantError._INVARIANTS; "
                "the allowlist is class-level closed by the metadata-only contract "
                "(DECISIONS.md#016)."
            )

    def __init__(self, *, event_id: UUID, invariant: str) -> None:
        # Literal class reference defends against runtime attribute injection;
        # SHA-256 prefix defends against the gate itself leaking the rejected value.
        if invariant not in AuditPersisterSchemaInvariantError._INVARIANTS:
            digest = hashlib.sha256(invariant.encode("utf-8", errors="replace")).hexdigest()[:12]
            raise ValueError(
                f"invariant must be one of "
                f"{sorted(AuditPersisterSchemaInvariantError._INVARIANTS)}; "
                f"got value with sha256-prefix={digest!r}, length={len(invariant)}."
            )
        super().__init__(f"{invariant} for event_id={event_id}; schema invariant violated")
        self.event_id = event_id
        self.invariant = invariant


class AuditPersisterEventRequestFieldMismatchError(ValueError, metaclass=_FrozenAllowlistMeta):
    """`persist()` was called with `event.X != request.X` for a
    pass-through field that drives eval isolation, audit attribution,
    or replay correctness. Same threat model as ReviewIdMismatch but
    for non-review-id fields shared between LLMRequest and LLMCallEvent.

    Strict-keyword `field_name: str` constructor; field name must
    match an allowlisted identifier in `_CHECKED_FIELDS` or
    construction raises `ValueError`. The message intentionally does
    NOT render the disagreeing values — those may be content-bearing
    (e.g., `context_summary` carries scope-unit paths). Operators
    query the DB to see the actual values; the exception names only
    the field.
    """

    _FROZEN_ALLOWLIST_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"_CHECKED_FIELDS", "_CANONICAL_RECOMPUTATION_FIELDS", "_FROZEN_ALLOWLIST_NAMES"}
    )

    _CHECKED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "is_eval",
            "node_id",
            "context_summary",
            "prompt_template_version",
            "degraded_mode",
            # §0b: pair the bool with its typed reason on the cross-check
            # allowlist so wrapper drift (event drops the reason, request
            # had one) is caught at persist time, not just at construction.
            "degradation_reason",
            "prompt_hash",
            "system_prompt_hash",
        }
    )

    # Fields whose comparison target is a canonical recomputation from the
    # request's prompt text, not a direct attribute on `LLMRequest`. Naming
    # `request.prompt_hash` in the error message would mislead — that field
    # does not exist on the request side. Included in `_FROZEN_ALLOWLIST_NAMES`
    # so parent-class reassignment is blocked by the metaclass `__setattr__`.
    _CANONICAL_RECOMPUTATION_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"prompt_hash", "system_prompt_hash"}
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        forbidden = {"_CHECKED_FIELDS", "_CANONICAL_RECOMPUTATION_FIELDS"}
        for name in forbidden:
            if name in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} cannot override "
                    f"AuditPersisterEventRequestFieldMismatchError.{name}; "
                    "the allowlist is class-level closed by the metadata-only contract "
                    "(DECISIONS.md#016)."
                )

    def __init__(self, *, field_name: str) -> None:
        cls = AuditPersisterEventRequestFieldMismatchError
        if field_name not in cls._CHECKED_FIELDS:
            digest = hashlib.sha256(field_name.encode("utf-8", errors="replace")).hexdigest()[:12]
            raise ValueError(
                f"field_name must be one of {sorted(cls._CHECKED_FIELDS)}; "
                f"got value with sha256-prefix={digest!r}, length={len(field_name)}."
            )
        if field_name in cls._CANONICAL_RECOMPUTATION_FIELDS:
            comparison = (
                f"event.{field_name} disagrees with the canonical hash "
                "recomputed from request prompts"
            )
        else:
            comparison = f"event.{field_name} disagrees with request.{field_name}"
        super().__init__(
            f"persist() called with mismatched {field_name}: {comparison}. "
            "Producer must build LLMCallEvent fields from LLMRequest (or "
            "vice versa). Persister refuses to attribute across diverging scopes."
        )
        self.field_name = field_name


class AuditPersisterEventResponseFieldMismatchError(ValueError, metaclass=_FrozenAllowlistMeta):
    """`persist()` was called with `event.X` disagreeing with the value
    the provider returned on `LLMResponse` (or with the canonical value
    derived from the response). Pass-through fields shared between
    `LLMResponse` and `LLMCallEvent` (model, token counts, latency,
    cache_hit) drive replay; recomputed fields (`cost_usd` via
    `compute_cost_usd`; `pricing_version` via the module constant)
    drive cost accounting after retention purges content.

    Same threat model as EventRequestFieldMismatch but for the
    provider-return-through side rather than the request-pass-through
    side. The two classes are kept separate so the exception name
    pinpoints which boundary diverged.
    """

    _FROZEN_ALLOWLIST_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"_CHECKED_FIELDS", "_CANONICAL_RECOMPUTATION_FIELDS", "_FROZEN_ALLOWLIST_NAMES"}
    )

    _CHECKED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "model",
            "input_tokens",
            "output_tokens",
            "latency_ms",
            "cached_tokens",
            "cache_hit",
            "cost_usd",
            "pricing_version",
        }
    )

    # Fields whose comparison target is a canonical recomputation, not a
    # direct attribute on `LLMResponse`. `cost_usd` derives from the
    # pricing table + response token counts; `pricing_version` is the
    # module constant. Included in `_FROZEN_ALLOWLIST_NAMES` so
    # parent-class reassignment is blocked by the metaclass.
    _CANONICAL_RECOMPUTATION_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"cost_usd", "pricing_version"}
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        forbidden = {"_CHECKED_FIELDS", "_CANONICAL_RECOMPUTATION_FIELDS"}
        for name in forbidden:
            if name in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} cannot override "
                    f"AuditPersisterEventResponseFieldMismatchError.{name}; "
                    "the allowlist is class-level closed by the metadata-only contract "
                    "(DECISIONS.md#016)."
                )

    def __init__(self, *, field_name: str) -> None:
        cls = AuditPersisterEventResponseFieldMismatchError
        if field_name not in cls._CHECKED_FIELDS:
            digest = hashlib.sha256(field_name.encode("utf-8", errors="replace")).hexdigest()[:12]
            raise ValueError(
                f"field_name must be one of {sorted(cls._CHECKED_FIELDS)}; "
                f"got value with sha256-prefix={digest!r}, length={len(field_name)}."
            )
        if field_name in cls._CANONICAL_RECOMPUTATION_FIELDS:
            comparison = (
                f"event.{field_name} disagrees with the canonical value "
                "recomputed from LLMResponse + pricing table"
            )
        else:
            comparison = f"event.{field_name} disagrees with the value returned on LLMResponse"
        super().__init__(
            f"persist() called with mismatched {field_name}: {comparison}. "
            "Producer must build LLMCallEvent metrics from the LLMResponse the "
            "provider actually returned. Persister refuses to attribute across "
            "diverging scopes."
        )
        self.field_name = field_name


# Spec-defined name; "Conflict" is the semantic category, not "Error".
class AuditPersisterIdempotencyConflict(ValueError):  # noqa: N818
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
      Pin test: `tests/unit/test_audit_persister.py
      ::test_compute_content_field_digests_intentionally_omits_non_text_columns`.

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
    AuditPersisterEventRequestFieldMismatchError,
    AuditPersisterEventResponseFieldMismatchError,
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


def _serialize_event_payload(
    event: (
        LLMCallEvent
        | ReviewPhaseEvent
        | FileExaminationEvent
        | AgentTransitionEvent
        | FindingEvent
        | FindingProposalRejectedEvent
        | AnalyzeResponseRejectedEvent
        | AnalyzeCompletedEvent
        | PublishRoutingEvent
        | PublishEligibilityEvent
        | PublishAttemptEvent
        | PublishEvent
    ),
) -> dict[str, Any]:
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
    """Durable persister; implements `LLMExchangePersister` + `PhaseEventSink`
    + `FileExaminationSink` + `AnalyzeEventSink` + `PublishEventSink`.

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
        # Event and request must agree on review_id. Otherwise the
        # persister would lookup installation_id via event.review_id
        # but store request's content under that installation —
        # misattributing the audit trail across review scopes.
        if request.review_id != event.review_id:
            raise AuditPersisterReviewIdMismatchError(
                event_review_id=event.review_id,
                request_review_id=request.review_id,
            )

        # Pass-through fields shared between LLMRequest and LLMCallEvent
        # must agree. The persister stores content from request while
        # taking audit metadata from event; mismatch means rows would
        # land under the wrong eval / attribution / replay scope.
        _direct_check_fields = AuditPersisterEventRequestFieldMismatchError._CHECKED_FIELDS - {
            "prompt_hash",
            "system_prompt_hash",
        }
        for field_name in _direct_check_fields:
            if getattr(event, field_name) != getattr(request, field_name):
                raise AuditPersisterEventRequestFieldMismatchError(field_name=field_name)

        # Event hashes must match canonical hashes of the request's
        # prompts. If they disagree, the audit row carries hash-of-X
        # while the content row holds text-Y; after retention purges
        # the content, only the (wrong) hash survives — replay would
        # reconstruct under a false identity.
        if event.prompt_hash != _canonical_prompt_hash(
            system_prompt=request.system_prompt, user_prompt=request.user_prompt
        ):
            raise AuditPersisterEventRequestFieldMismatchError(field_name="prompt_hash")
        if event.system_prompt_hash != _canonical_system_prompt_hash(request.system_prompt):
            raise AuditPersisterEventRequestFieldMismatchError(field_name="system_prompt_hash")

        # Provider-return-through fields shared between LLMResponse and
        # LLMCallEvent must agree. Split into two groups:
        #
        # (1) STABLE fields (model, token counts, latency, cache state)
        #     are checked pre-tx — disagreement is a producer bug
        #     regardless of when the call landed. Pre-tx avoids a
        #     wasted transaction on clearly malformed pairs.
        #
        # (2) PRICING-version-bound fields (`cost_usd`, `pricing_version`)
        #     are checked INSIDE the transaction, only on the fresh-write
        #     branch. `compute_cost_usd` reads the current pricing table
        #     and `PRICING_VERSION` is the current module constant; both
        #     change over deploys. An idempotent re-emission of an event
        #     originally persisted under an older pricing version would
        #     legitimately carry the old `pricing_version` / `cost_usd`,
        #     and the audit-conflict path is the right verifier for that
        #     case (payload equality against the stored row). Catching
        #     pricing drift pre-tx would block those re-emissions.
        _stable_response_fields = (
            AuditPersisterEventResponseFieldMismatchError._CHECKED_FIELDS
            - AuditPersisterEventResponseFieldMismatchError._CANONICAL_RECOMPUTATION_FIELDS
        )
        _response_value_for = {
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "cached_tokens": response.cache_read_tokens,
            "cache_hit": response.cache_read_tokens > 0,
        }
        for field_name in _stable_response_fields:
            if getattr(event, field_name) != _response_value_for[field_name]:
                raise AuditPersisterEventResponseFieldMismatchError(field_name=field_name)

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

            # Fresh-write pricing cross-check. Reachable only on the
            # freshly-inserted audit branch — never on idempotent re-emit.
            # An older event re-emitted across a `PRICING_VERSION` bump
            # carries its original `cost_usd` / `pricing_version`; the
            # audit-conflict path above is the right verifier for that
            # case. Pre-tx pricing checks would block such re-emits.
            # Brand-new events MUST use the current pricing snapshot;
            # raising here rolls back the freshly-inserted audit row.
            canonical_cost_usd = float(
                compute_cost_usd(
                    response.model,
                    input_tokens=response.input_tokens,
                    cache_write_tokens=response.cache_write_tokens,
                    cache_read_tokens=response.cache_read_tokens,
                    output_tokens=response.output_tokens,
                )
            )
            if event.cost_usd != canonical_cost_usd:
                raise AuditPersisterEventResponseFieldMismatchError(field_name="cost_usd")
            if event.pricing_version != PRICING_VERSION:
                raise AuditPersisterEventResponseFieldMismatchError(field_name="pricing_version")

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

    # -- FileExaminationSink surface ----------------------------------------

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        """Persist a FileExaminationEvent row to audit_events.

        Mirrors `emit_phase` semantics: idempotent on `event.event_id`;
        payload-mismatch on PK conflict raises `AuditPersisterIdempotencyConflict`;
        no content side-table (FileExaminationEvent carries only structural
        identifiers — file_path, examination_type, parse_status, skip_reason —
        none of which is content per `DECISIONS.md#014` point 5's borderline-
        fields rule).

        `phase_key` is written as NULL (the denormalized top-level column is
        populated only for `ReviewPhaseEvent`; per the `_NO_PHASE_KEY` sentinel
        rule, every other event type writes NULL).

        Intake's phase-2 content fan-out emits these concurrently under
        `asyncio.TaskGroup`; each emission opens its own `AsyncSession` so
        the fan-out is safe under the per-call session discipline shared
        with `emit_phase`.
        """
        await self._persist_non_phase_event(event)

    # -- AnalyzeEventSink surface -------------------------------------------

    async def _persist_non_phase_event(
        self,
        event: (
            FileExaminationEvent
            | FindingEvent
            | FindingProposalRejectedEvent
            | AnalyzeResponseRejectedEvent
            | AnalyzeCompletedEvent
            | PublishRoutingEvent
            | PublishEligibilityEvent
            | PublishAttemptEvent
            | PublishEvent
        ),
    ) -> None:
        """Persist any non-phase audit event row to audit_events.

        Shared body for `FileExaminationSink` + `AnalyzeEventSink`
        emit_* methods — every event whose `phase_key` is NULL. Mirrors
        `emit_phase`'s idempotency + payload-mismatch discipline.

        Per-call session discipline: each emission opens its own
        `AsyncSession` (no concurrent reuse). Safe under the V1.5
        parallel-analyze fan-out per the `phase-events-bound-work`
        sibling-sink rule.
        """
        payload = _serialize_event_payload(event)

        async with self._session_factory() as session, session.begin():
            stmt = (
                postgresql_insert(AuditEventRow)
                .values(
                    event_id=event.event_id,
                    review_id=event.review_id,
                    event_type=event.event_type,
                    phase_key=_NO_PHASE_KEY,
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

    async def emit_finding(self, event: FindingEvent) -> None:
        """Persist a `FindingEvent` row to audit_events (analyze admitted finding)."""
        await self._persist_non_phase_event(event)

    async def emit_finding_proposal_rejected(self, event: FindingProposalRejectedEvent) -> None:
        """Persist a `FindingProposalRejectedEvent` row (parser rejection)."""
        await self._persist_non_phase_event(event)

    async def emit_analyze_response_rejected(self, event: AnalyzeResponseRejectedEvent) -> None:
        """Persist an `AnalyzeResponseRejectedEvent` row (response-level parse failure)."""
        await self._persist_non_phase_event(event)

    async def emit_analyze_completed(self, event: AnalyzeCompletedEvent) -> None:
        """Persist an `AnalyzeCompletedEvent` row (per-pass aggregate)."""
        await self._persist_non_phase_event(event)

    # -- PublishEventSink surface -------------------------------------------
    # Per DECISIONS.md #023 (publish routing and eligibility are separate
    # decisions): four publish-emitted event types share one Protocol because
    # they form one logical group (per-finding routing + eligibility + the
    # terminal per-attempt outcome + the review-level summary) and one
    # transaction-lifecycle discipline.

    async def emit_publish_routing(self, event: PublishRoutingEvent) -> None:
        """Persist a `PublishRoutingEvent` row (per-finding routing decision)."""
        await self._persist_non_phase_event(event)

    async def emit_publish_eligibility(self, event: PublishEligibilityEvent) -> None:
        """Persist a `PublishEligibilityEvent` row (per-finding policy gate)."""
        await self._persist_non_phase_event(event)

    async def emit_publish_attempt(self, event: PublishAttemptEvent) -> None:
        """Persist a `PublishAttemptEvent` row (terminal GitHub-call outcome)."""
        await self._persist_non_phase_event(event)

    async def emit_publish_result(self, event: PublishEvent) -> None:
        """Persist a `PublishEvent` row (success-path review-level summary)."""
        await self._persist_non_phase_event(event)

    async def query_prior_publish_event(self, review_id: UUID) -> PublishEvent | None:
        """Return the most-recent prior `PublishEvent` for `review_id`, or None.

        Per FUP-064: the V1 publish node's intra-Outrider idempotency
        pre-flight check. A same-`review_id` redispatch (dispatcher
        re-fires the webhook after agent crash + restart) hits this
        query BEFORE the GitHub call; on hit, the publish node short-
        circuits to `idempotently_skipped` outcome without burning a
        GitHub round-trip.

        Read-only — opens its own `AsyncSession` (no `session.begin()`,
        no transaction needed for a single SELECT). Mirrors the
        per-emit session discipline so the persister stays
        concurrent-safe across reviews.

        Multi-row handling (replay re-emission divergence): returns the
        most-recent by `timestamp` via `ORDER BY timestamp DESC LIMIT 1`.
        The append-only audit log can legitimately carry multiple
        `PublishEvent` rows for one `review_id` (per Q5 withdrawal:
        replay re-emission produces additional rows; consumer-side
        dedup keys off `(review_id, github_review_id)`). This query
        chooses the most-recent row as the canonical "did we publish";
        consumer-side drift surfaces via V1.5 anomaly rules (FUP-063),
        not this method.

        Deserialization: the JSONB payload round-trips through
        `PublishEvent.model_validate(payload)` — the event's frozen
        + extra=forbid + validator chain re-fires, so a corrupted
        payload raises `ValidationError` at the read boundary rather
        than producing a silently-wrong return value.
        """
        # The discriminator string MUST match PublishEvent's event_type
        # Literal default. A future rename of the Literal value (e.g.,
        # "publish" → "publish_result") would silently disable this
        # query if the filter were a hardcoded string. Pulled via
        # `model_fields[...].default` so the magic string lives in one
        # place: the PublishEvent schema declaration.
        publish_event_type: str = PublishEvent.model_fields["event_type"].default
        async with self._session_factory() as session:
            stmt = (
                select(AuditEventRow.payload)
                .where(
                    AuditEventRow.review_id == review_id,
                    AuditEventRow.event_type == publish_event_type,
                )
                .order_by(AuditEventRow.timestamp.desc())
                .limit(1)
            )
            payload = await session.scalar(stmt)
        if payload is None:
            return None
        return PublishEvent.model_validate(payload)
