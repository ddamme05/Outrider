# See specs/2026-05-16-audit-persister.md + DECISIONS.md#014/#016.
"""AuditPersister — durable single-class implementation of eight sink Protocols.

Implements `LLMExchangePersister` (`llm/base.py`) AND `PhaseEventSink`,
`FileExaminationSink`, `AnalyzeEventSink`, `PublishEventSink`,
`TraceEventSink`, `HITLEventSink`, `SynthesizeEventSink`
(`audit/sinks.py`) from one body,
sharing transaction lifecycle and session-per-call discipline. The
non-phase events route through a shared `_persist_non_phase_event`
helper so the idempotency + payload-mismatch discipline is uniform
across event types; natural-key-mode events
(`TraceDecisionEvent`, `HITLRequestEvent`, `HITLDecisionEvent`) route
through `_persist_keyed_by_natural_key` per `DECISIONS.md#026`. The
PublishEventSink surface was added per DECISIONS.md #023 (publish
routing and eligibility are separate decisions); HITLEventSink + the
generalized natural-key dispatch landed per
`specs/2026-05-26-hitl-node.md`.

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

import asyncio
import hashlib
import json
import time
from collections.abc import (  # noqa: TC003  (runtime use in @dataclass field annotation)
    AsyncIterator,
    Callable,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, NamedTuple

# Runtime import: typed-kwarg signatures on the strict-keyword exception
# constructors reference UUID as a parameter annotation. TYPE_CHECKING-only
# would only suffice if annotations were string-quoted; runtime-validated
# parameter shapes benefit from a real type at module import.
from uuid import UUID  # noqa: TC003 — runtime annotation needed for typed exception constructors

from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

from outrider.db.models.audit_events import AuditEvent as AuditEventRow
from outrider.db.models.findings import Finding
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
        CacheLookupEvent,
        CacheServeEvent,
        FileExaminationEvent,
        FindingProposalRejectedEvent,
        LLMCallEvent,
        PublishAttemptEvent,
        PublishEligibilityEvent,
        PublishRoutingEvent,
        ReviewPhaseEvent,
        ScopeExclusionEvent,
        SynthesizeCompletedEvent,
    )
    from outrider.llm.base import LLMRequest, LLMResponse
    from outrider.schemas.review_finding import ReviewFinding

# PublishEvent is consumed at RUNTIME by `query_prior_publish_event`'s
# `PublishEvent.model_validate(payload)` call — must be imported at
# module level, not under TYPE_CHECKING. TraceDecisionEvent likewise:
# `_persist_keyed_by_natural_key`'s no-op recovery path calls
# `TraceDecisionEvent.model_validate(existing_payload)` to reconstruct
# the canonical persisted event for the M7 (b) return contract.
from outrider.audit.aggregates import (  # noqa: E402  (runtime: read-side query delegate)
    ReviewLLMAggregates,
    aggregate_review_llm_metrics,
)
from outrider.audit.events import (
    FindingEvent,  # noqa: E402  (constructed at runtime by _lift_finding_event)
    HITLDecisionEvent,  # noqa: E402  (model_validate at runtime)
    HITLRequestEvent,  # noqa: E402  (model_validate at runtime)
    PublishEvent,  # noqa: E402  (intentional post-TYPE_CHECKING runtime import)
    ReplayVerdictEvent,  # noqa: E402  (model_validate at runtime in query_replay_verdict_event)
    TraceDecisionEvent,  # noqa: E402  (model_validate at runtime)
)

__all__ = [
    "AuditPersister",
    "AuditPersisterConfigError",
    "AuditPersisterEventRequestFieldMismatchError",
    "AuditPersisterEventResponseFieldMismatchError",
    "AuditPersisterFindingInstallationIdMismatchError",
    "AuditPersisterIsEvalMismatchError",
    "AuditPersisterHITLDecisionIdempotencyLookupError",
    "AuditPersisterHITLDecisionNaturalKeyConflict",
    "AuditPersisterHITLRequestIdempotencyLookupError",
    "AuditPersisterHITLRequestNaturalKeyConflict",
    "AuditPersisterIdempotencyConflict",
    "AuditPersisterNaturalKeyConflict",
    "AuditPersisterNaturalKeyLookupError",
    "AuditPersisterPublishLockAcquisitionTimeoutError",
    "AuditPersisterReviewIdMismatchError",
    "AuditPersisterReviewNotFoundError",
    "AuditPersisterSchemaInvariantError",
    "AuditPersisterTraceIdempotencyLookupError",
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

# The `replay_verdict` partial-index predicate as LITERAL SQL, never an
# ORM expression. The ON CONFLICT arbiter only matches the partial unique
# index `uq_audit_events_replay_verdict_natural_key` when `index_where` is
# literal text identical to the migration's `... WHERE event_type =
# 'replay_verdict'`. An ORM expression (`AuditEventRow.event_type == "..."`)
# renders a bind parameter (`event_type = $1`), which psycopg3 generic plans
# can't prove implies the index's constant predicate, so arbiter inference
# fails (42P10) once the statement is server-prepared. Mirrors the literal
# `sa_text(...)` form the natural-key path uses for the same reason.
_REPLAY_VERDICT_INDEX_WHERE: Final = sa_text("event_type = 'replay_verdict'")

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


class AuditPersisterFindingInstallationIdMismatchError(ValueError):
    """`emit_finding()` was called with `finding.installation_id` disagreeing
    with the `installation_id` resolved from the finding's `reviews` row.

    The reviews row is the FK-scope source of truth: `findings.installation_id`
    is `ON DELETE RESTRICT` to `installations` and drives purge scoping, so a
    finding whose own `installation_id` diverges from its review's would write
    a content row under a fabricated scope. Fail loud rather than persist it.

    Strict-keyword `finding_installation_id: int` + `review_installation_id: int`
    + `review_id: UUID` constructor; message generated from the three
    identifiers (metadata-only, no finding content).
    """

    def __init__(
        self,
        *,
        finding_installation_id: int,
        review_installation_id: int,
        review_id: UUID,
    ) -> None:
        super().__init__(
            f"emit_finding() called with mismatched installation scope: "
            f"finding.installation_id={finding_installation_id} but "
            f"reviews.installation_id={review_installation_id} for "
            f"review_id={review_id}. The reviews row is the FK-scope source "
            "of truth; persister refuses to attribute a finding across "
            "installation scopes."
        )
        self.finding_installation_id = finding_installation_id
        self.review_installation_id = review_installation_id
        self.review_id = review_id


class AuditPersisterIsEvalMismatchError(ValueError):
    """An event was persisted with `event.is_eval` disagreeing with the
    `is_eval` resolved from its `reviews` row.

    Eval isolation depends on every row a review touches carrying that review's
    single `is_eval` value (`docs/testing.md` "Eval isolation"). Two FUP-130
    defenses enforce it: (1) this WRITE-SIDE guard fails loud when a producer
    emits a divergent `is_eval` at the two content-bearing sites that resolve the
    reviews row — `persist()` (`LLMCallEvent`) and `emit_finding()`
    (`FindingEvent`); and (2) the dashboard READ-API filters every per-review
    event/content read by `is_eval == reviews.is_eval` (the read-side defense
    that closes the leak at its manifestation point, mirroring replay's
    `_verify_is_eval_consistent`). This exception is the write-side half — the
    `is_eval` twin of the persister's `installation_id` cross-check
    (`AuditPersisterFindingInstallationIdMismatchError`). Fail loud rather than
    persist the divergence.

    Scope of THIS write-side guard: only the two sites that already resolve the
    reviews row (resolving in non-resolving paths like `emit_phase` would cost an
    extra SELECT per event). The dashboard read-side predicates — not this guard —
    are what cover the surfaces those non-resolving paths feed (synthesize
    metrics, publish/HITL lifecycle, policy_version, the events explorer).

    Strict-keyword `event_is_eval: bool` + `review_is_eval: bool` + `review_id:
    UUID` constructor; message is metadata-only (two booleans + a UUID).
    """

    def __init__(
        self,
        *,
        event_is_eval: bool,
        review_is_eval: bool,
        review_id: UUID,
    ) -> None:
        super().__init__(
            f"event persisted with mismatched eval scope: "
            f"event.is_eval={event_is_eval} but reviews.is_eval={review_is_eval} "
            f"for review_id={review_id}. Every row a review touches must carry "
            "the review's is_eval; persister refuses to cross eval/production "
            "isolation."
        )
        self.event_is_eval = event_is_eval
        self.review_is_eval = review_is_eval
        self.review_id = review_id


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
            # `_persist_keyed_by_natural_key` is a polymorphic helper
            # over `{TraceDecisionEvent, HITLRequestEvent,
            # HITLDecisionEvent}`. The narrowing wrappers
            # (`emit_trace_decision`, `emit_hitl_request`,
            # `emit_hitl_decision`) assert the registry returned the
            # matching event_type for the input. A return-type
            # mismatch is a registry-routing bug; the per-method
            # static identifier here lets operators pinpoint which
            # wrapper saw the mismatch via the audit log's event_id.
            # Dynamic `type(result).__name__` is intentionally NOT
            # part of the identifier (allowlist requires static
            # strings); operators correlate event_id → audit row →
            # observed event_type to recover that information.
            "emit_trace_decision return-type mismatch",
            "emit_hitl_request return-type mismatch",
            "emit_hitl_decision return-type mismatch",
            # `emit_phase`'s natural-key conflict path scans for an
            # existing row via the same predicate the partial unique
            # index targets; the index reported a conflict but the
            # SELECT found nothing. Either the partial-index predicate
            # diverged from the SELECT's predicate (schema regression)
            # or a concurrent DELETE raced (impossible under append-
            # only — see DECISIONS.md#016). Operator triages via
            # event_id → audit row → diff index predicate vs SELECT
            # predicate in `emit_phase`.
            "natural-key conflict on review_phase but no matching row found on reload",
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
            # FUP-096: the constrained-decoding provenance must survive the
            # wrapper intact — event digest diverging from the request's
            # derived digest would mislabel the output population replay
            # and the cache telemetry split on.
            "response_format_digest",
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


# Spec-defined name; "Conflict" is the semantic category, not "Error".
class AuditPersisterNaturalKeyConflict(ValueError):  # noqa: N818
    """Natural-key re-emission with diverging identity-subset payload.

    Sibling of `AuditPersisterIdempotencyConflict` for the natural-key
    idempotency mode per `DECISIONS.md#026`. Raised by
    `_persist_keyed_by_natural_key` when the partial unique index fires
    `on_conflict_do_nothing` AND the follow-up SELECT returns an
    existing row whose identity-subset fields disagree with the
    incoming event.

    Polymorphic over event_type per the natural-key registry:

      - trace_decision: `natural_key=(("source_finding_id", str(uuid)),)`
      - hitl_request:   `natural_key=()` (review_id alone)
      - hitl_decision:  `natural_key=()` (review_id alone)

    The PK-conflict sibling carries a single `event_id` — under
    `event_id`-PK conflict the conflicting event id IS the lookup key.
    Under natural-key conflict the operator needs BOTH the existing
    row's PK (to pull it for inspection) AND the incoming PK (to trace
    the producing call), so this exception carries both.

    Metadata-only by contract per `DECISIONS.md#016` point 4. Attributes:

    - `existing_event_id`: the PK of the row already in `audit_events`.
    - `incoming_event_id`: the PK the persister attempted to insert.
    - `review_id`: the natural-key tuple's first component (always).
    - `natural_key`: tuple of `(field_name, str_value)` pairs for the
      JSONB-payload components of the natural key (empty tuple when the
      key is `review_id` alone). Schema identifiers + stringified values.
    - `mismatched_fields`: tuple of identity-subset field names whose
      values differ between existing and incoming. Schema identifiers
      only — never values.

    `source_finding_id` accessor (deprecated, trace-only):
      callers/tests that grew up around the trace-specific shape can
      still read `exc.source_finding_id` when the natural key carries
      `source_finding_id`; it raises `AttributeError` for HITL events
      so any test that misroutes to a HITL conflict surfaces loudly.

    Operators investigating a natural-key conflict pull both
    `audit_events` rows out-of-band (by `event_id`) and compare full
    payloads; this exception is the signal, the DB is the source.
    """

    def __init__(
        self,
        *,
        existing_event_id: UUID,
        incoming_event_id: UUID,
        review_id: UUID,
        natural_key: tuple[tuple[str, str], ...] = (),
        mismatched_fields: tuple[str, ...],
    ) -> None:
        # An empty mismatched_fields tuple paired with this exception is
        # an internal bug — the helper should NOT raise on identity-subset
        # equality. Fail loud on construction.
        if not mismatched_fields:
            raise ValueError(
                "AuditPersisterNaturalKeyConflict requires non-empty "
                "mismatched_fields; an empty tuple paired with this "
                "exception class would describe identity-subset equality, "
                "which is the no-op recovery path (return existing event), "
                "not a conflict. Caller bug."
            )
        self.existing_event_id = existing_event_id
        self.incoming_event_id = incoming_event_id
        self.review_id = review_id
        self.natural_key = natural_key
        self.mismatched_fields = mismatched_fields
        # Trace-style display: include the natural-key tuple inline when
        # present, otherwise just `(review_id=...)`.
        key_repr = (
            f"(review_id={review_id})"
            if not natural_key
            else (f"(review_id={review_id}, " + ", ".join(f"{k}={v}" for k, v in natural_key) + ")")
        )
        super().__init__(
            f"natural-key conflict on {key_repr}: "
            f"existing_event_id={existing_event_id}, "
            f"incoming_event_id={incoming_event_id}, "
            f"mismatched_fields={mismatched_fields}"
        )

    @property
    def source_finding_id(self) -> UUID:
        """Trace-specific accessor preserved for backward compatibility.

        Returns the `source_finding_id` UUID if `natural_key` carries it
        (trace-decision events). Raises `AttributeError` otherwise so a
        test that accidentally reads this attribute on a HITL conflict
        fails loudly rather than silently returning a misleading value.
        """
        for field_name, value in self.natural_key:
            if field_name == "source_finding_id":
                return UUID(value)
        raise AttributeError(
            "source_finding_id is not part of this natural-key conflict "
            f"(natural_key={self.natural_key!r}); the field is "
            "trace-decision-specific."
        )


class AuditPersisterHITLRequestNaturalKeyConflict(AuditPersisterNaturalKeyConflict):
    """Natural-key conflict on `(review_id) WHERE event_type='hitl_request'`.

    Subclass of `AuditPersisterNaturalKeyConflict` for distinct operator
    triage. The natural-key is `review_id` alone (per V1 single-shot
    per-review semantics); `natural_key=()` on this subclass.
    Identity-subset divergence on `findings_requiring_approval` /
    `auto_post_findings` / `created_at` / `expires_at` / `is_eval` raises
    this exception — surfaces a producer-side derivation drift
    (e.g., `state.received_at` mutation or `HITL_TIMEOUT_MINUTES` config
    change between body invocations).
    """


class AuditPersisterHITLDecisionNaturalKeyConflict(AuditPersisterNaturalKeyConflict):
    """Natural-key conflict on `(review_id) WHERE event_type='hitl_decision'`.

    Subclass of `AuditPersisterNaturalKeyConflict` for distinct operator
    triage. The natural-key is `review_id` alone (V1 single-shot per
    Non-goals "No multi-gate semantics within a single review");
    `natural_key=()` on this subclass. Identity-subset divergence on
    `decisions_content_hash` / `is_eval` raises this exception — the
    canonical signal for the divergent-content concurrent-decide race
    the dashboard endpoint's wrapper catches as a no-op.
    """


# See DECISIONS.md#027 — V1 per-review publish-side advisory lock.
class AuditPersisterPublishLockAcquisitionTimeoutError(TimeoutError):  # noqa: N818
    """Raised by `acquire_publish_lock` when bounded try-lock retries exhaust.

    Bounded by `max_wait_seconds` from the first probe attempt. Indicates
    either a slow legitimate holder (long GitHub POST under contention)
    or a pathologically-hung holder. The publish node's outer try/except
    wrapping `enter_async_context` catches this exception and emits
    `PublishAttemptEvent(outcome=failed, failure_class="AuditPersister"
    "PublishLockAcquisitionTimeoutError")` BEFORE re-raising, honoring
    the node's raises contract.

    Carries `review_id` + `waited_seconds` for operator triage:
    `review_id` identifies the contested review; `waited_seconds` names
    the timeout bound that was hit (NOT the actual wait — the loop's
    deadline check fires once `monotonic() >= start + max_wait`, so
    waited_seconds reflects the configured bound, not a precise stopwatch).
    """

    def __init__(self, *, review_id: UUID, waited_seconds: int) -> None:
        self.review_id = review_id
        self.waited_seconds = waited_seconds
        # Message renders only `review_id` (UUID identifier — class-level
        # allowlisted in the metadata-only test). `waited_seconds` is
        # stored as an instance attribute for programmatic access but
        # deliberately NOT interpolated into the message string: the
        # `test_every_metadata_only_exception_type_is_actually_metadata_only`
        # gate injects a forbidden-content sentinel for any constructor
        # parameter whose stringified annotation doesn't match its
        # allowlist (UUID/tuple/Mapping/str-allowlisted/int-class).
        # Under `from __future__ import annotations` the int annotation
        # is stringified as `"int"`, which doesn't satisfy the test's
        # `ann is int` identity check — so the parameter would be
        # treated as content-bearing if interpolated. Keeping the bound
        # off the str/repr/args channels preserves the contract;
        # operators read `exc.waited_seconds` directly.
        super().__init__(
            f"Could not acquire publish lock for review {review_id} — "
            f"possible slow holder under contention or hung-process "
            f"scenario. See exception attribute `waited_seconds` for "
            f"the configured timeout bound."
        )


# Module-private sentinel for subclass-only construction of
# `AuditPersisterNaturalKeyLookupError`. Replaces a prior `_from_subclass: bool`
# kwarg — a boolean kwarg could be passed from external callers
# (`Cls("attacker text", _from_subclass=True)`) bypassing the protection;
# a sentinel object is constructible only inside this module.
_AUDIT_PERSISTER_INTERNAL_TOKEN = object()


class AuditPersisterNaturalKeyLookupError(LookupError):
    """Generic base for natural-key conflict-but-empty-SELECT errors.

    The `on_conflict_do_nothing` path returns "no rows" on either
    (a) conflict — existing row blocks insert; follow-up SELECT loads it
    and identity-subset comparison proceeds — OR (b) follow-up SELECT
    actually returns zero rows. Case (b) should never happen under V1
    (single-threaded per review + append-only `audit_events`), but
    defense-in-depth matters for V1.5 parallel-analyze + concurrent
    webhook redispatch + future audit-archive flows. Subclasses
    specialize per event_type so operators can differentiate which
    natural-key index fired empty.

    Strict-keyword `message` per the project-wide metadata-only
    persister-exception contract: callers cannot pass arbitrary
    positional content into the exception's `args[0]`. Subclasses
    construct their own message from review_id-shaped identifiers and
    pass it through this base, gated by the module-private sentinel
    `_AUDIT_PERSISTER_INTERNAL_TOKEN` — boolean gates are bypassable
    by external callers passing `_from_subclass=True`, sentinel-based
    gates are not (the sentinel object identity cannot be replicated
    from outside the module).
    """

    def __init__(self, message: str, *, _token: object = None) -> None:
        if _token is not _AUDIT_PERSISTER_INTERNAL_TOKEN:
            raise TypeError(
                "AuditPersisterNaturalKeyLookupError must be constructed by "
                "a subclass that passes the module-private "
                "`_AUDIT_PERSISTER_INTERNAL_TOKEN` sentinel; direct "
                "construction would let arbitrary content land in args[0], "
                "breaking the metadata-only persister-exception contract."
            )
        super().__init__(message)


class AuditPersisterTraceIdempotencyLookupError(AuditPersisterNaturalKeyLookupError):
    """Natural-key conflict fired but follow-up SELECT returned no row.

    The `postgresql_insert(...).on_conflict_do_nothing(...)` path
    returns "no rows" in two cases:
    (a) conflict — the existing row blocks the insert; the follow-up
        SELECT loads it and the identity-subset comparison proceeds.
    (b) follow-up SELECT actually returns zero rows.

    Case (b) should not happen under V1 (single-threaded per review,
    `audit_events` is append-only — no DELETE path), but defense-in-depth
    matters for V1.5 parallel-analyze + concurrent webhook redispatch
    + future audit-archive flows. Distinct exception type so operators
    can differentiate "trace's natural-key lookup failed mid-flight"
    from generic DB transport errors.

    Strict-keyword `review_id` + `source_finding_id` constructor; the
    message is generated from those identifiers.
    """

    def __init__(self, *, review_id: UUID, source_finding_id: UUID) -> None:
        # noqa S608 on each f-string fragment — string is an exception
        # message, not a SQL query (the word "SELECT" appears in the
        # diagnostic prose, which trips Bandit's regex).
        message = (
            f"_persist_keyed_by_natural_key: on-conflict path fired but "  # noqa: S608
            f"follow-up SELECT on (review_id={review_id}, "  # noqa: S608
            f"source_finding_id={source_finding_id}) returned no row. "  # noqa: S608
            "Either the row was concurrently removed (audit append-only "
            "trigger should prevent this) or the natural-key SELECT "
            "predicate diverged from the partial unique index expression."
        )
        super().__init__(message, _token=_AUDIT_PERSISTER_INTERNAL_TOKEN)
        self.review_id = review_id
        self.source_finding_id = source_finding_id


class AuditPersisterHITLRequestIdempotencyLookupError(AuditPersisterNaturalKeyLookupError):
    """HITL-request natural-key conflict fired but follow-up SELECT returned no row.

    Natural-key is `(review_id) WHERE event_type='hitl_request'`. Same
    defense-in-depth shape as the trace lookup error — should not happen
    under V1, but distinct exception type lets operators triage on event
    type rather than on natural-key shape.
    """

    def __init__(self, *, review_id: UUID) -> None:
        super().__init__(
            f"_persist_keyed_by_natural_key: hitl_request natural-key "  # noqa: S608
            f"conflict fired but follow-up SELECT on (review_id={review_id}, "  # noqa: S608
            "event_type='hitl_request') returned no row.",  # noqa: S608
            _token=_AUDIT_PERSISTER_INTERNAL_TOKEN,
        )
        self.review_id = review_id


class AuditPersisterHITLDecisionIdempotencyLookupError(AuditPersisterNaturalKeyLookupError):
    """HITL-decision natural-key conflict fired but follow-up SELECT returned no row.

    Natural-key is `(review_id) WHERE event_type='hitl_decision'`. Sibling
    of the request-side lookup error; same shape, distinct event_type.
    """

    def __init__(self, *, review_id: UUID) -> None:
        super().__init__(
            f"_persist_keyed_by_natural_key: hitl_decision natural-key "  # noqa: S608
            f"conflict fired but follow-up SELECT on (review_id={review_id}, "  # noqa: S608
            "event_type='hitl_decision') returned no row.",  # noqa: S608
            _token=_AUDIT_PERSISTER_INTERNAL_TOKEN,
        )
        self.review_id = review_id


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
    AuditPersisterFindingInstallationIdMismatchError,
    AuditPersisterIsEvalMismatchError,
    AuditPersisterIdempotencyConflict,
    AuditPersisterNaturalKeyConflict,
    AuditPersisterHITLRequestNaturalKeyConflict,
    AuditPersisterHITLDecisionNaturalKeyConflict,
    AuditPersisterPublishLockAcquisitionTimeoutError,
    AuditPersisterNaturalKeyLookupError,
    AuditPersisterTraceIdempotencyLookupError,
    AuditPersisterHITLRequestIdempotencyLookupError,
    AuditPersisterHITLDecisionIdempotencyLookupError,
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


_TRACE_DECISION_IDENTITY_SUBSET: Final[frozenset[str]] = frozenset(
    {
        "source_finding_id",
        "target_file",
        "resolution_status",
        "is_eval",
    }
)

# HITL natural-key identity subsets per the HITL spec:
#   - hitl_request: both `created_at` and `expires_at` are stable Q4
#     derivations from `state.received_at`; including both surfaces drift
#     in EITHER recipe loudly as a persister conflict.
#   - hitl_decision: `decisions_content_hash` is the canonical
#     content-derived digest; `reviewer_id` is EXCLUDED (advisory under
#     V1 single-tenant scope per Non-goals — same logical decision under
#     different reviewer_ids should collapse, not raise).
_HITL_REQUEST_IDENTITY_SUBSET: Final[frozenset[str]] = frozenset(
    {
        "findings_requiring_approval",
        "auto_post_findings",
        "created_at",
        "expires_at",
        "is_eval",
    }
)

_HITL_DECISION_IDENTITY_SUBSET: Final[frozenset[str]] = frozenset(
    {
        "decisions_content_hash",
        "is_eval",
    }
)

# Fields whose payload value is set-semantic and must be sorted before
# identity-equality. JSON serialization preserves list order, so without
# this canonicalization a retry whose tuple ordering shuffled would
# compare unequal against the existing row even though the SET of values
# is identical. Per DECISIONS.md#026 the trace_decision identity subset
# does NOT include `resolved_candidate_paths` (LLM-ranking-order variance
# via its `proposed_import_strings` derivation defeats lockstep), so the
# only set-semantic identity fields here are HITLRequestEvent's pair.
_SET_SEMANTIC_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        # HITLRequestEvent — both tuples are set-semantic per the
        # `HITLRequest._enforce_finding_partition` validator (each
        # finding appears at most once across both). The HITL node
        # canonicalizes via `tuple(sorted(...))` on emit; this
        # persister-side guard is defense-in-depth so a future producer
        # change that drops sorting doesn't trigger spurious
        # `AuditPersisterHITLRequestNaturalKeyConflict` raises.
        "findings_requiring_approval",
        "auto_post_findings",
    }
)

# Registry shape so the extension surface is obvious: adding a second
# natural-key event type is a single mapping entry (plus a partial unique
# index + an `emit_*` method). The MappingProxyType wrapper blocks
# post-import mutation — callers cannot widen the registry by
# `_IDENTITY_SUBSETS["x"] = frozenset()` and silently admit a wrong-mode
# write.
_IDENTITY_SUBSETS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "trace_decision": _TRACE_DECISION_IDENTITY_SUBSET,
        "hitl_request": _HITL_REQUEST_IDENTITY_SUBSET,
        "hitl_decision": _HITL_DECISION_IDENTITY_SUBSET,
    }
)


@dataclass(frozen=True)
class _NaturalKeySpec:
    """Per-event-type natural-key shape for `_persist_keyed_by_natural_key`.

    Each entry pins:
      - `event_type`: the literal stored in `audit_events.event_type`,
        matching the migration's partial-index WHERE clause.
      - `jsonb_key`: the JSONB payload field extending the natural key
        beyond `review_id` (e.g. trace's `source_finding_id`); `None`
        means the natural key is `review_id` alone.
      - `event_class`: for `model_validate(existing_payload)` recovery
        on the no-op path.
      - `conflict_cls`: distinct subclass of
        `AuditPersisterNaturalKeyConflict` for operator triage.
      - `build_lookup_error`: factory the helper calls when the
        `on_conflict_do_nothing` path returns no row AND the follow-up
        SELECT also returns no row — distinct per event_type.
    """

    event_type: str
    jsonb_key: str | None
    event_class: type[Any]
    conflict_cls: type[AuditPersisterNaturalKeyConflict]
    build_lookup_error: Callable[[Any], AuditPersisterNaturalKeyLookupError]


def _build_trace_lookup_error(event: Any) -> AuditPersisterNaturalKeyLookupError:
    return AuditPersisterTraceIdempotencyLookupError(
        review_id=event.review_id,
        source_finding_id=event.source_finding_id,
    )


def _build_hitl_request_lookup_error(event: Any) -> AuditPersisterNaturalKeyLookupError:
    return AuditPersisterHITLRequestIdempotencyLookupError(review_id=event.review_id)


def _build_hitl_decision_lookup_error(event: Any) -> AuditPersisterNaturalKeyLookupError:
    return AuditPersisterHITLDecisionIdempotencyLookupError(review_id=event.review_id)


_NATURAL_KEY_SPECS: Final[Mapping[str, _NaturalKeySpec]] = MappingProxyType(
    {
        "trace_decision": _NaturalKeySpec(
            event_type="trace_decision",
            jsonb_key="source_finding_id",
            event_class=TraceDecisionEvent,
            conflict_cls=AuditPersisterNaturalKeyConflict,
            build_lookup_error=_build_trace_lookup_error,
        ),
        "hitl_request": _NaturalKeySpec(
            event_type="hitl_request",
            jsonb_key=None,
            event_class=HITLRequestEvent,
            conflict_cls=AuditPersisterHITLRequestNaturalKeyConflict,
            build_lookup_error=_build_hitl_request_lookup_error,
        ),
        "hitl_decision": _NaturalKeySpec(
            event_type="hitl_decision",
            jsonb_key=None,
            event_class=HITLDecisionEvent,
            conflict_cls=AuditPersisterHITLDecisionNaturalKeyConflict,
            build_lookup_error=_build_hitl_decision_lookup_error,
        ),
    }
)


def _canonicalize_for_identity_compare(field: str, value: Any) -> Any:
    """Normalize set-semantic identity fields before equality.

    For fields in `_SET_SEMANTIC_IDENTITY_FIELDS`, sort the value (if it's
    a list-shaped JSON payload value) so two retries with the same set but
    different emission order compare equal. `_MISSING` and non-list values
    pass through unchanged.
    """
    if field in _SET_SEMANTIC_IDENTITY_FIELDS and isinstance(value, list):
        return sorted(value)
    return value


def _payload_identity_subset(event_type: str) -> frozenset[str]:
    """Identity-subset field names for natural-key idempotency comparison.

    Returns the set of payload field names whose values are compared
    between an incoming event and an existing row when the natural-key
    partial unique index fires `on_conflict_do_nothing`. Equality across
    the subset = legitimate retry (return existing row's event); any
    divergence = real conflict (raise `AuditPersisterNaturalKeyConflict`).

    For `trace_decision` per `DECISIONS.md#026` (point 3) +
    `specs/2026-05-23-trace-node.md` M7 (c): `{source_finding_id,
    target_file, resolution_status, is_eval}`.

      - `source_finding_id`: the natural-key payload component (the
        index's other lookup column is `review_id`, pinned by the
        natural-key index lookup at SELECT time).
      - `target_file`: deterministic resolution outcome of the resolver
        applied to the project tree — divergence means the tree changed,
        which IS a real conflict.
      - `resolution_status`: deterministic outcome class
        (`resolved`/`unresolved`/`ambiguous`).
      - `is_eval`: invariant per review; cross-retry divergence is a
        config bug.

    Deliberately EXCLUDES (each would defeat the audit-first contract
    on legitimate retries): `event_id`, `timestamp`, `sequence_number`
    (already excluded by `_serialize_event_payload`), `review_id`
    (pinned by index lookup), `reason` (LLM-narrative noise),
    `proposed_import_strings` / `resolved_candidate_paths` / `trace_path`
    (per-emission noise — `resolved_candidate_paths` is derived from
    LLM-ranking-order-variant `proposed_import_strings` per #026 point 3),
    `event_type` (pinned by partial-index WHERE).

    V1 supports `trace_decision`, `hitl_request`, and `hitl_decision`
    — the natural-key mode per `DECISIONS.md#026` is shared across
    these three event types (the trace-arc landed it first; the HITL
    arc generalized the dispatch via the `_NATURAL_KEY_SPECS`
    registry). Unknown event types are a producer bug: the caller
    routed to the natural-key helper for an event type the persister
    doesn't know how to compare. Fail loud on unknown event_type to
    surface the routing bug at the persister boundary rather than
    silently admitting a wrong-mode write.
    """
    try:
        return _IDENTITY_SUBSETS[event_type]
    except KeyError:
        raise ValueError(
            f"_payload_identity_subset: unsupported event_type={event_type!r}; "
            f"natural-key idempotency mode is only defined for "
            f"{sorted(_IDENTITY_SUBSETS)} in V1 (per DECISIONS.md#026)."
        ) from None


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
        | ScopeExclusionEvent
        | CacheLookupEvent
        | CacheServeEvent
        | TraceDecisionEvent
        | HITLRequestEvent
        | HITLDecisionEvent
        | SynthesizeCompletedEvent
        | ReplayVerdictEvent
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
# Findings content-writer helpers (specs/2026-05-30-findings-content-writer.md).
# ---------------------------------------------------------------------------


def _lift_finding_event(finding: ReviewFinding, *, is_eval: bool) -> FindingEvent:
    """Lift an admitted `ReviewFinding` to its metadata-only `FindingEvent`.

    The audit row stays metadata-only per DECISIONS.md#014; the human-
    readable content (title/description/evidence) goes to the `findings`
    content row, never to the audit payload. Absorbed from the former
    `analyze._lift_admitted_finding` so the persister is the single write
    authority for both rows.
    """
    return FindingEvent(
        review_id=finding.review_id,
        is_eval=is_eval,
        finding_id=finding.finding_id,
        finding_type=finding.finding_type,
        severity=finding.severity,
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        dimension=finding.dimension,
        finding_content_hash=finding.content_hash,
        evidence_tier=finding.evidence_tier,
        query_match_id=finding.query_match_id,
        trace_path=finding.trace_path,
        policy_version=finding.policy_version,
        proposal_hash=finding.proposal_hash,
    )


def _normalize_trace_path(value: Any) -> list[Any] | None:
    """Coerce a `trace_path` (tuple from the finding, list from the DB) to a
    comparable list; None stays None."""
    if value is None:
        return None
    return list(value)


# The analyze-time-immutable verify set: every `findings` column written at
# analyze time and never mutated after. `_finding_verify_values` builds the
# {column -> incoming-value} mapping that the re-emit/conflict paths compare
# against the stored row. EXCLUDES `publish_destination` + the override quartet
# (`original_severity`/`override_reason`/`overrider_id`) — read-model projection
# columns that are NULL in V1 (no post-HITL findings writer exists: publish/HITL
# mutate the in-memory `ReviewFinding` + emit audit events, never UPDATE the
# row; the audit stream is canonical per DECISIONS.md#034). They are excluded so
# that IF a future denormalized writer populates them, a re-emit carrying NULL
# does not false-raise against the populated row.
def _finding_verify_values(
    finding: ReviewFinding,
    *,
    is_eval: bool,
    installation_id: int,
) -> dict[str, Any]:
    """The incoming {column -> value} mapping for the verify set.

    Enum-bearing columns render as their `.value`; `trace_path` normalizes
    tuple↔list; `installation_id` is the value resolved from the reviews row
    (not the mutable `ReviewFinding` field).
    """
    return {
        "content_hash": finding.content_hash,
        "is_eval": is_eval,
        "installation_id": installation_id,
        "finding_type": finding.finding_type.value,
        "dimension": finding.dimension.value,
        "severity": finding.severity.value,
        "evidence_tier": finding.evidence_tier.value,
        "file_path": finding.file_path,
        "line_start": finding.line_start,
        "line_end": finding.line_end,
        "title": finding.title,
        "description": finding.description,
        "evidence": finding.evidence,
        "suggested_fix": finding.suggested_fix,
        "query_match_id": finding.query_match_id,
        "policy_version": finding.policy_version,
        "trace_path": _normalize_trace_path(finding.trace_path),
    }


def _finding_row_db_value(db_row: Any, column: str) -> Any:
    """Read `column` from the stored `findings` row, normalizing `trace_path`
    tuple↔list so the comparison matches the incoming side."""
    if column == "trace_path":
        return _normalize_trace_path(db_row.trace_path)
    return getattr(db_row, column)


def _finding_row_mismatches(
    db_row: Any,
    finding: ReviewFinding,
    *,
    is_eval: bool,
    installation_id: int,
) -> tuple[str, ...]:
    """Names of analyze-time-immutable `findings` columns whose stored value
    disagrees with the incoming finding. Empty tuple = match."""
    expected = _finding_verify_values(finding, is_eval=is_eval, installation_id=installation_id)
    return tuple(
        col
        for col, new_value in expected.items()
        if _finding_row_db_value(db_row, col) != new_value
    )


def _finding_field_digests(
    db_row: Any,
    finding: ReviewFinding,
    mismatched: tuple[str, ...],
    *,
    is_eval: bool,
    installation_id: int,
) -> Mapping[str, FieldDigest]:
    """`FieldDigest` (SHA-256 + byte-length of each side) per mismatched column.

    Digests only — content (title/description/evidence) never reaches logs.
    Mirrors `_compute_field_digests`; keys are a subset of `mismatched`, which
    satisfies the `AuditPersisterIdempotencyConflict` subset invariant.
    """
    expected = _finding_verify_values(finding, is_eval=is_eval, installation_id=installation_id)
    digests: dict[str, FieldDigest] = {}
    for col in mismatched:
        existing_text = repr(_finding_row_db_value(db_row, col))
        attempted_text = repr(expected[col])
        digests[col] = FieldDigest(
            existing_sha256=_sha256_text(existing_text),
            attempted_sha256=_sha256_text(attempted_text),
            existing_length=len(existing_text.encode("utf-8")),
            attempted_length=len(attempted_text.encode("utf-8")),
        )
    return digests


# ---------------------------------------------------------------------------
# AuditPersister.
# ---------------------------------------------------------------------------


class AuditPersister:
    """Durable persister; implements `LLMExchangePersister` + `PhaseEventSink`
    + `FileExaminationSink` + `AnalyzeEventSink` + `PublishEventSink`
    + `TraceEventSink` + `HITLEventSink` + `SynthesizeEventSink`.

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
            # Step 1: resolve installation_id + is_eval from the reviews row.
            # This row is created upstream (webhook handler) before graph
            # dispatch; its absence is a producer-side bug.
            review_row = (
                await session.execute(
                    select(Review.installation_id, Review.is_eval).where(
                        Review.id == event.review_id
                    )
                )
            ).one_or_none()
            if review_row is None:
                raise AuditPersisterReviewNotFoundError(review_id=event.review_id)
            installation_id, review_is_eval = review_row
            # FUP-130: the reviews row's is_eval is the source of truth. Fail
            # loud here (write-side guard) on a divergent event; the dashboard
            # read-API also filters these llm_call rows by is_eval (read-side
            # defense). Sibling of emit_finding's installation_id cross-check.
            if event.is_eval != review_is_eval:
                raise AuditPersisterIsEvalMismatchError(
                    event_is_eval=event.is_eval,
                    review_is_eval=review_is_eval,
                    review_id=event.review_id,
                )

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

        Idempotent on the natural key `(review_id, phase_id,
        COALESCE(phase_key, ''), marker)` per the partial unique index
        `uq_audit_events_review_phase_natural_key`. The HITL node body
        re-emits `marker='start'` on every resume (LangGraph durable-
        execution restarts the body from the top); without natural-key
        idempotency, fresh `event_id`s would land duplicate `start`
        rows on every resume — defeating `phase-events-bound-work`
        replay tooling that treats the start/end pair as a single
        causal barrier per node entry. The deterministic `phase_id`
        from `compute_phase_id(...)` is the producer-side half of the
        contract; this `on_conflict_do_nothing` is the consumer-side
        gate.

        Populates the top-level denormalized `phase_key` column from
        `event.phase_key` (typically `None` in V1; V1.5 parallel-analyze
        will populate per-file). V1.5's per-file index queries depend on
        this column being populated correctly today; the natural-key
        index's `COALESCE(phase_key, '')` expression makes the dedup
        work for both V1 (all-NULL) and V1.5 (per-file string) rows.
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
                .on_conflict_do_nothing(
                    index_elements=[
                        AuditEventRow.review_id,
                        sa_text("(payload->>'phase_id')"),
                        sa_text("COALESCE(phase_key, '')"),
                        sa_text("(payload->>'marker')"),
                    ],
                    # Literal text MUST match the migration's
                    # `CREATE UNIQUE INDEX ... WHERE ...` predicate
                    # exactly so the planner targets the partial index.
                    # Drift between this string and the migration would
                    # fall through to a sequential conflict scan or skip
                    # the index entirely. No SQL-injection surface: all
                    # tokens are literal.
                    index_where=sa_text(
                        "event_type = 'review_phase' "
                        "AND payload ? 'phase_id' "
                        "AND payload ? 'marker'"
                    ),
                )
                .returning(AuditEventRow.event_id)
            )
            inserted_event_id = await session.scalar(stmt)
            if inserted_event_id is None:
                # Natural-key conflict fired (the partial unique index
                # matched on `(review_id, phase_id, COALESCE(phase_key,
                # ''), marker)`). Load the existing row and compare the
                # non-ephemeral, non-natural-key fields. Per CodeRabbit
                # 2026-05-27: phase_id is a `str` field so a producer
                # bug could mint a phase_id that collides with a real
                # row but carry different `node_id` or `is_eval` —
                # without this reload+compare the producer bug
                # silently no-ops instead of surfacing as a loud
                # conflict. The reload reads by the SAME predicate
                # the index targets (review_id + JSONB-extracted
                # phase_id + COALESCE(phase_key, '') + JSONB-extracted
                # marker) so it deterministically resolves to the
                # single existing row.
                existing_payload = await session.scalar(
                    select(AuditEventRow.payload).where(
                        AuditEventRow.review_id == event.review_id,
                        AuditEventRow.event_type == "review_phase",
                        AuditEventRow.payload["phase_id"].astext == event.phase_id,
                        sa_func.coalesce(AuditEventRow.phase_key, "") == (event.phase_key or ""),
                        AuditEventRow.payload["marker"].astext == event.marker,
                    )
                )
                if existing_payload is None:
                    # Index reported conflict but our reload found
                    # nothing — schema invariant violation. The
                    # partial-index predicate diverged from this
                    # query's predicate, OR a concurrent DELETE
                    # raced the SELECT (impossible under append-only).
                    raise AuditPersisterSchemaInvariantError(
                        event_id=event.event_id,
                        invariant=(
                            "natural-key conflict on review_phase but "
                            "no matching row found on reload"
                        ),
                    )
                # Compare non-ephemeral fields. `event_id` + `timestamp`
                # are per-emission and excluded. The natural-key fields
                # (`phase_id`, `marker`, `phase_key`) match by index
                # definition. The remaining payload fields are
                # `node_id`. `is_eval` is a top-level column, not
                # payload, so include it in the existing-row load
                # explicitly via a second SELECT or fold into payload
                # comparison — for emit_phase the simpler shape is
                # JSONB-only compare on `node_id`, with `is_eval`
                # cross-check via the existing row's column.
                # Build a single-field digest map for each compare to
                # keep `field_digests` keys aligned with `mismatched_fields`
                # (constructor invariant: digest keys ⊆ mismatched fields).
                # Per-emission noise (`event_id`, `timestamp`) and natural-
                # key fields (already matched by definition) are excluded.
                if existing_payload.get("node_id") != event.node_id:
                    raise AuditPersisterIdempotencyConflict(
                        event_id=event.event_id,
                        mismatched_fields=("node_id",),
                        field_digests={
                            "node_id": FieldDigest(
                                existing_sha256=_value_or_missing_sha256(
                                    existing_payload.get("node_id", _MISSING)
                                ),
                                attempted_sha256=_value_or_missing_sha256(event.node_id),
                                existing_length=_value_or_missing_length(
                                    existing_payload.get("node_id", _MISSING)
                                ),
                                attempted_length=_value_or_missing_length(event.node_id),
                            )
                        },
                    )
                existing_is_eval = await session.scalar(
                    select(AuditEventRow.is_eval).where(
                        AuditEventRow.review_id == event.review_id,
                        AuditEventRow.event_type == "review_phase",
                        AuditEventRow.payload["phase_id"].astext == event.phase_id,
                        sa_func.coalesce(AuditEventRow.phase_key, "") == (event.phase_key or ""),
                        AuditEventRow.payload["marker"].astext == event.marker,
                    )
                )
                if existing_is_eval != event.is_eval:
                    # `is_eval` is a top-level column, not a payload field;
                    # the digest is computed against the bool directly.
                    raise AuditPersisterIdempotencyConflict(
                        event_id=event.event_id,
                        mismatched_fields=("is_eval",),
                        field_digests={
                            "is_eval": FieldDigest(
                                existing_sha256=_value_or_missing_sha256(existing_is_eval),
                                attempted_sha256=_value_or_missing_sha256(event.is_eval),
                                existing_length=_value_or_missing_length(existing_is_eval),
                                attempted_length=_value_or_missing_length(event.is_eval),
                            )
                        },
                    )

    # -- FileExaminationSink surface ----------------------------------------

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        """Persist a FileExaminationEvent row to audit_events.

        Idempotent on `event.event_id` via the `event_id`-PK
        `on_conflict_do_nothing` path inside `_persist_non_phase_event`.
        Payload-mismatch on PK conflict raises
        `AuditPersisterIdempotencyConflict`. No content side-table —
        FileExaminationEvent carries only structural identifiers
        (file_path, examination_type, parse_status, skip_reason) and
        none of those is content per `DECISIONS.md#014` point 5's
        borderline-fields rule.

        Distinct idempotency mode from `emit_phase`: per
        `DECISIONS.md#026` (and the `uq_audit_events_review_phase_
        natural_key` partial unique index added 2026-05-27),
        `emit_phase` dedupes on the natural key `(review_id, phase_id,
        COALESCE(phase_key, ''), marker)` because the HITL node body
        re-emits phase events with fresh `event_id`s on resume. File-
        examination emissions have no body-replay path that mints
        fresh event_ids for the same logical row, so the simpler
        `event_id`-PK dedup remains correct here.

        `phase_key` is written as NULL (the denormalized top-level
        column is populated only for `ReviewPhaseEvent`; per the
        `_NO_PHASE_KEY` sentinel rule, every other event type writes
        NULL).

        Intake's phase-2 content fan-out emits these concurrently
        under `asyncio.TaskGroup`; each emission opens its own
        `AsyncSession` so the fan-out is safe under the per-call
        session discipline shared with `emit_phase`.
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
            | ScopeExclusionEvent
            | CacheLookupEvent
            | CacheServeEvent
            | SynthesizeCompletedEvent
        ),
    ) -> None:
        """Persist any non-phase audit event row to audit_events.

        Shared body for `FileExaminationSink` + `AnalyzeEventSink`
        emit_* methods — every event whose `phase_key` is NULL.
        Idempotent on `event.event_id` via the `event_id`-PK
        `on_conflict_do_nothing` path below; on conflict, the existing
        row's payload is loaded and compared against the incoming
        payload — mismatch raises `AuditPersisterIdempotencyConflict`,
        match returns silently. `emit_phase` deliberately diverges from
        this discipline as of 2026-05-27: phase events use natural-key
        idempotency on `(review_id, phase_id, COALESCE(phase_key, ''),
        marker)` because the HITL node body re-emits phase events with
        fresh `event_id`s on resume (LangGraph durable execution
        restarts the body from the top). Non-phase events have no
        body-replay path that mints fresh event_ids for the same
        logical row, so event_id-PK dedup remains correct here.

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

    async def emit_finding(self, finding: ReviewFinding, *, is_eval: bool) -> None:
        """Co-insert the `FindingEvent` audit row + the `findings` content row.

        One transaction, mirroring the DECISIONS.md#016 `LLMCallEvent` +
        `llm_call_content` co-insert. The audit row stays metadata-only per
        DECISIONS.md#014; the `findings` row carries the human-readable
        content (title/description/evidence) on the content tier.

        The content-write decision keys on `finding_id`, NOT `event_id`: a
        mid-node analyze retry can append a second `FindingEvent` (fresh
        `event_id`) for the same `finding_id`, and the `findings` row must not
        be re-inserted (no-resurrection guard) once the content has purged.
        `installation_id` is resolved from the reviews row (the trustworthy
        FK-scope source), then cross-checked against the finding's own value.
        See specs/2026-05-30-findings-content-writer.md.
        """
        event = _lift_finding_event(finding, is_eval=is_eval)
        payload = _serialize_event_payload(event)

        async with self._session_factory() as session, session.begin():
            # Step 0: resolve installation_id + is_eval from the reviews row.
            # Absence is a producer-side bug — the reviews row exists before
            # graph dispatch.
            review_row = (
                await session.execute(
                    select(Review.installation_id, Review.is_eval).where(
                        Review.id == finding.review_id
                    )
                )
            ).one_or_none()
            if review_row is None:
                raise AuditPersisterReviewNotFoundError(review_id=finding.review_id)
            installation_id, review_is_eval = review_row

            # Cross-check the finding's own installation_id against the
            # reviews-row source of truth. A disagreement means the producer
            # attributed the finding to the wrong installation scope; fail loud
            # rather than write a content row under a fabricated scope. The
            # typed exception keeps this on the persister's metadata-only
            # exception taxonomy (METADATA_ONLY_EXCEPTION_TYPES) rather than a
            # raw ValueError, so the wrapper's str(exc) safety contract holds.
            if finding.installation_id != installation_id:
                raise AuditPersisterFindingInstallationIdMismatchError(
                    finding_installation_id=finding.installation_id,
                    review_installation_id=installation_id,
                    review_id=finding.review_id,
                )
            # FUP-130: the is_eval twin of the installation_id cross-check above.
            # The reviews row's is_eval is the source of truth; fail loud here
            # (write-side guard) on a divergent event. The dashboard findings
            # read also filters by is_eval (read-side defense).
            if event.is_eval != review_is_eval:
                raise AuditPersisterIsEvalMismatchError(
                    event_is_eval=event.is_eval,
                    review_is_eval=review_is_eval,
                    review_id=finding.review_id,
                )

            retention_expires_at = event.timestamp + self._retention_settings.findings_retention_ttl

            # Step 1: INSERT the audit row, ON CONFLICT no-op on event_id.
            audit_stmt = (
                postgresql_insert(AuditEventRow)
                .values(
                    event_id=event.event_id,
                    review_id=event.review_id,
                    event_type=event.event_type,
                    phase_key=_NO_PHASE_KEY,  # FindingEvent has no phase_key
                    timestamp=event.timestamp,
                    is_eval=event.is_eval,
                    payload=payload,
                )
                .on_conflict_do_nothing(index_elements=["event_id"])
                .returning(AuditEventRow.event_id)
            )
            inserted_audit = await session.scalar(audit_stmt)
            audit_row_already_existed = inserted_audit is None

            # Step 3 (Case A/B): determine re-emit vs first-emit, keyed on
            # finding_id. The content-write decision below uses this flag.
            if audit_row_already_existed:
                # Case A — same-event_id retry. Unconditionally a re-emit.
                # First verify the stored audit payload matches (drift = bug).
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
                is_reemit = True
            else:
                # Case B — audit row freshly inserted. A re-emit only if some
                # OTHER FindingEvent already carries this finding_id (a prior
                # fresh-event_id emit reached the persister). Reads a JSONB
                # payload field (not indexed today — flag a partial index on
                # (payload->>'finding_id') for FindingEvent rows if hot).
                count = await session.scalar(
                    select(sa_func.count())
                    .select_from(AuditEventRow)
                    .where(
                        AuditEventRow.event_type == "finding",
                        AuditEventRow.payload["finding_id"].astext == str(finding.finding_id),
                        AuditEventRow.event_id != event.event_id,
                    )
                )
                is_reemit = (count or 0) > 0

            if is_reemit:
                # Re-emit branch: SELECT the findings row by finding_id over the
                # analyze-time-immutable verify set. None → no-op (the
                # no-resurrection guard: never re-INSERT purged content, firing
                # for both Case A and Case B). Exists → verify or raise.
                row = (
                    await session.execute(
                        select(
                            Finding.content_hash,
                            Finding.is_eval,
                            Finding.installation_id,
                            Finding.finding_type,
                            Finding.dimension,
                            Finding.severity,
                            Finding.evidence_tier,
                            Finding.file_path,
                            Finding.line_start,
                            Finding.line_end,
                            Finding.title,
                            Finding.description,
                            Finding.evidence,
                            Finding.suggested_fix,
                            Finding.query_match_id,
                            Finding.trace_path,
                            Finding.policy_version,
                        ).where(Finding.finding_id == finding.finding_id)
                    )
                ).one_or_none()
                if row is None:
                    return  # purged content — respect retention, no resurrection
                mismatched = _finding_row_mismatches(
                    row, finding, is_eval=is_eval, installation_id=installation_id
                )
                if mismatched:
                    raise AuditPersisterIdempotencyConflict(
                        event_id=event.event_id,
                        mismatched_fields=mismatched,
                        field_digests=_finding_field_digests(
                            row,
                            finding,
                            mismatched,
                            is_eval=is_eval,
                            installation_id=installation_id,
                        ),
                    )
                return

            # First-emit branch: INSERT the findings row, with
            # on_conflict_do_nothing(["finding_id"]) as the concurrent-writer
            # net, then SELECT-verify-or-noop on the no-row-returned path.
            content_stmt = (
                postgresql_insert(Finding)
                .values(
                    finding_id=finding.finding_id,
                    review_id=finding.review_id,
                    installation_id=installation_id,
                    policy_version=finding.policy_version,
                    finding_type=finding.finding_type.value,
                    dimension=finding.dimension.value,
                    severity=finding.severity.value,
                    evidence_tier=finding.evidence_tier.value,
                    file_path=finding.file_path,
                    line_start=finding.line_start,
                    line_end=finding.line_end,
                    title=finding.title,
                    description=finding.description,
                    evidence=finding.evidence,
                    suggested_fix=finding.suggested_fix,
                    query_match_id=finding.query_match_id,
                    trace_path=(
                        list(finding.trace_path) if finding.trace_path is not None else None
                    ),
                    content_hash=finding.content_hash,
                    is_eval=is_eval,
                    retention_expires_at=retention_expires_at,
                )
                .on_conflict_do_nothing(index_elements=["finding_id"])
                .returning(Finding.finding_id)
            )
            inserted = await session.scalar(content_stmt)
            if inserted is not None:
                return

            # Conflict: a concurrent writer landed the findings row between our
            # audit INSERT and this content INSERT. Verify-or-noop, identical
            # shape to the re-emit branch (purge between INSERT and SELECT wins
            # over conflict detection via one_or_none()).
            row = (
                await session.execute(
                    select(
                        Finding.content_hash,
                        Finding.is_eval,
                        Finding.installation_id,
                        Finding.finding_type,
                        Finding.dimension,
                        Finding.severity,
                        Finding.evidence_tier,
                        Finding.file_path,
                        Finding.line_start,
                        Finding.line_end,
                        Finding.title,
                        Finding.description,
                        Finding.evidence,
                        Finding.suggested_fix,
                        Finding.query_match_id,
                        Finding.trace_path,
                        Finding.policy_version,
                    ).where(Finding.finding_id == finding.finding_id)
                )
            ).one_or_none()
            if row is None:
                return  # purged between INSERT and SELECT; respect retention
            mismatched = _finding_row_mismatches(
                row, finding, is_eval=is_eval, installation_id=installation_id
            )
            if mismatched:
                raise AuditPersisterIdempotencyConflict(
                    event_id=event.event_id,
                    mismatched_fields=mismatched,
                    field_digests=_finding_field_digests(
                        row,
                        finding,
                        mismatched,
                        is_eval=is_eval,
                        installation_id=installation_id,
                    ),
                )

    async def emit_finding_proposal_rejected(self, event: FindingProposalRejectedEvent) -> None:
        """Persist a `FindingProposalRejectedEvent` row (parser rejection)."""
        await self._persist_non_phase_event(event)

    async def emit_analyze_response_rejected(self, event: AnalyzeResponseRejectedEvent) -> None:
        """Persist an `AnalyzeResponseRejectedEvent` row (response-level parse failure)."""
        await self._persist_non_phase_event(event)

    async def emit_analyze_completed(self, event: AnalyzeCompletedEvent) -> None:
        """Persist an `AnalyzeCompletedEvent` row (per-pass aggregate)."""
        await self._persist_non_phase_event(event)

    async def emit_scope_exclusion(self, event: ScopeExclusionEvent) -> None:
        """Persist a `ScopeExclusionEvent` row (per-file trivial-scope
        classification; event_id-PK idempotent per `DECISIONS.md#026`)."""
        await self._persist_non_phase_event(event)

    async def emit_cache_lookup(self, event: CacheLookupEvent) -> None:
        """Persist a `CacheLookupEvent` row (analyze-cache shadow
        telemetry; event_id-PK idempotent per `DECISIONS.md#026`)."""
        await self._persist_non_phase_event(event)

    async def emit_cache_serve(self, event: CacheServeEvent) -> None:
        """Persist a `CacheServeEvent` row (analyze-cache serve-flip stage;
        event_id-PK idempotent per `DECISIONS.md#026`)."""
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

    # -- SynthesizeEventSink surface ----------------------------------------
    # Per pre-spec gate #1: synthesize uses event_id-PK idempotency
    # (NOT natural-key). The natural-key state-lockstep gate iii fails
    # because `ReviewReport.summary` text lives in `llm_call_content`,
    # not in the audit-row payload — a natural-key persister couldn't
    # return enough payload to reconstruct ReviewReport on retry.
    # Mirrors `AnalyzeCompletedEvent` shape (event_id-PK,
    # `_persist_non_phase_event` body).

    async def emit_synthesize_completed(self, event: SynthesizeCompletedEvent) -> None:
        """Persist a `SynthesizeCompletedEvent` row (per-review aggregate)."""
        await self._persist_non_phase_event(event)

    async def emit_replay_verdict(self, event: ReplayVerdictEvent) -> bool:
        """Append a `ReplayVerdictEvent`, idempotent on `(review_id)` for
        `event_type='replay_verdict'` via the partial unique index
        `uq_audit_events_replay_verdict_natural_key`. Returns True if a verdict was
        newly inserted, False if one already existed (the idempotent no-op) — so the
        projector counts only fresh projections even under concurrent ticks (mirrors
        the `.returning()`-and-branch shape of `_persist_keyed_by_natural_key`).

        Deliberately NOT the natural-key divergence-detecting path (`emit_hitl_*`):
        a replay verdict is DETERMINISTIC over the append-only judged prefix, so a
        re-projection produces the identical row and `on_conflict_do_nothing` (no
        content comparison) is the right altitude — unlike a HITL decision, two
        verdicts for one review cannot legitimately diverge. If a projector bug ever
        produced a divergent verdict the first-written one wins and the read side
        surfaces it. Per-call `AsyncSession` in its own transaction, like the other
        emit methods. The `index_where` mirrors the migration's partial-index WHERE
        clause exactly, or the conflict-arbiter falls through to a seq scan.
        """
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
                    payload=_serialize_event_payload(event),
                )
                .on_conflict_do_nothing(
                    index_elements=[AuditEventRow.review_id],
                    index_where=_REPLAY_VERDICT_INDEX_WHERE,
                )
                .returning(AuditEventRow.event_id)
            )
            inserted = await session.scalar(stmt)
        return inserted is not None

    # -- Natural-key idempotency surface ------------------------------------
    # Per `DECISIONS.md#026` natural-key idempotency mode. The shape supports
    # multiple event types (trace_decision, hitl_request, hitl_decision in
    # V1) via `_NATURAL_KEY_SPECS` below. Each spec names: the event class
    # (for `model_validate` reconstruction on the no-op path), the JSONB
    # natural-key extension (None for review-id-only keys; field name for
    # JSONB-extracted keys like trace's `source_finding_id`), the
    # conflict / lookup-error exception classes for distinct operator
    # triage. The helper returns the canonical persisted event so the
    # producer node can construct the state-layer schema in lockstep
    # with audit per the audit-first emission contract.

    async def _persist_keyed_by_natural_key(
        self, event: TraceDecisionEvent | HITLRequestEvent | HITLDecisionEvent
    ) -> TraceDecisionEvent | HITLRequestEvent | HITLDecisionEvent:
        """Persist a natural-key-keyed audit event under idempotency.

        Polymorphic over `event.event_type` via `_NATURAL_KEY_SPECS`.
        Insert via `postgresql_insert(...).on_conflict_do_nothing(...)`
        against the per-event-type partial unique index. Insert path:
        returns the incoming event verbatim. Conflict path: SELECTs the
        existing row, reconstructs an event from its payload, compares
        identity-subset per `_payload_identity_subset(event.event_type)`.
        Subset equal → return existing event (the lockstep-recovery
        winner). Subset diverges → raise the event_type-specific
        `AuditPersisterNaturalKeyConflict` subclass.

        The `index_where` predicate passed to `on_conflict_do_nothing`
        MUST exactly mirror the partial-index WHERE clause for SQLAlchemy
        to bind to the right index. Drift between the SELECT predicate
        + the index_where + the migration's `CREATE UNIQUE INDEX ... WHERE`
        would silently fall through to a full-table conflict-arbiter
        search and the on-conflict semantics would not fire.

        Raises:
          `AuditPersisterNaturalKeyLookupError` subclass: insert path
            returned no row AND follow-up natural-key SELECT also returned
            no row. Defends against the race / append-only-violation case.
          `AuditPersisterNaturalKeyConflict` subclass: insert path returned
            no row, follow-up SELECT loaded a row, identity-subset
            comparison detected at least one field divergence.
        """
        spec = _NATURAL_KEY_SPECS[event.event_type]
        payload = _serialize_event_payload(event)
        event_type_literal = spec.event_type
        # SELECT predicate matches the partial-index expression exactly.
        # For trace, the JSONB component is `(payload->>'source_finding_id')`;
        # for HITL it's None (review_id alone). Drift between this
        # predicate and the migration's `CREATE UNIQUE INDEX ... ON ...`
        # expression would make the planner fall through to a seq scan.
        if spec.jsonb_key is not None:
            jsonb_value: str = str(getattr(event, spec.jsonb_key))
            natural_key_select_predicate = (
                (AuditEventRow.review_id == event.review_id)
                & (AuditEventRow.event_type == event_type_literal)
                & (AuditEventRow.payload[spec.jsonb_key].astext == jsonb_value)
            )
            index_elements: list[Any] = [
                AuditEventRow.review_id,
                sa_text(f"(payload->>'{spec.jsonb_key}')"),
            ]
            index_where_clause = (
                f"event_type = '{event_type_literal}' AND payload ? '{spec.jsonb_key}'"
            )
            conflict_natural_key: tuple[tuple[str, str], ...] = ((spec.jsonb_key, jsonb_value),)
        else:
            # review_id alone — partial unique index ON (review_id)
            # WHERE event_type = '...'. No JSONB extraction.
            natural_key_select_predicate = (AuditEventRow.review_id == event.review_id) & (
                AuditEventRow.event_type == event_type_literal
            )
            index_elements = [AuditEventRow.review_id]
            index_where_clause = f"event_type = '{event_type_literal}'"
            conflict_natural_key = ()

        # Concurrent emits with identical natural-keys serialize via PG's
        # unique-index conflict resolution: one transaction wins INSERT;
        # the other gets `inserted_event_id=None` and follows the
        # conflict-path. V1 is single-threaded per review, but V1.5
        # parallel-analyze + webhook redispatch exercise this race
        # window — `session.begin()` + `on_conflict_do_nothing` is the
        # right shape.

        async with self._session_factory() as session, session.begin():
            insert_stmt = (
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
                .on_conflict_do_nothing(
                    index_elements=index_elements,
                    # `event_type_literal` is the Pydantic `Literal[...]`
                    # default from the schema — never user input. The
                    # f-string interpolation in `index_where_clause` is
                    # safe (no SQL-injection surface); we use it because
                    # SQLAlchemy's partial-index WHERE needs literal SQL
                    # text matching the migration's `CREATE UNIQUE INDEX
                    # ... WHERE event_type = '<lit>' ...` exactly.
                    index_where=sa_text(index_where_clause),
                )
                .returning(AuditEventRow.event_id)
            )
            inserted_event_id = await session.scalar(insert_stmt)
            if inserted_event_id is not None:
                # Insert-path: the incoming event IS the canonical persisted
                # event. Return it verbatim — caller builds the state-layer
                # schema from this exact event.
                return event

            # Conflict-path: load the existing row and compare identity-subset.
            existing_row = (
                await session.execute(
                    select(AuditEventRow.event_id, AuditEventRow.payload).where(
                        natural_key_select_predicate
                    )
                )
            ).one_or_none()
            if existing_row is None:
                raise spec.build_lookup_error(event)
            existing_event_id, existing_payload = existing_row

            # `_MISSING` sentinel distinguishes "field absent" from
            # "field present with value None". Bare `.get(field)` would
            # treat them as equal — a drifted payload that DROPPED a
            # nullable identity field (e.g. `target_file`) would compare
            # equal to one that legitimately has `target_file=None` and
            # get silently classified as a retry.
            mismatched = tuple(
                sorted(
                    field
                    for field in _payload_identity_subset(event.event_type)
                    if _canonicalize_for_identity_compare(
                        field, existing_payload.get(field, _MISSING)
                    )
                    != _canonicalize_for_identity_compare(field, payload.get(field, _MISSING))
                )
            )
            if mismatched:
                raise spec.conflict_cls(
                    existing_event_id=existing_event_id,
                    incoming_event_id=event.event_id,
                    review_id=event.review_id,
                    natural_key=conflict_natural_key,
                    mismatched_fields=mismatched,
                )

            # Identity-subset equal: legitimate retry. Reconstruct the
            # existing event and return it — producer builds the state-layer
            # schema from THIS event (original per-emission fields), keeping
            # state and audit in lockstep across retries even when the
            # incoming event's per-emission noise fields (e.g. trace's
            # `reason` / `proposed_import_strings`) differ from the
            # originally-persisted ones.
            # `spec.event_class` is `type[Any]` in the registry (the field is
            # bounded to natural-key event types but the dataclass storage
            # is generic); reconstruction returns the typed event.
            return spec.event_class.model_validate(existing_payload)  # type: ignore[no-any-return]

    async def emit_trace_decision(self, event: TraceDecisionEvent) -> TraceDecisionEvent:
        """Persist a TraceDecisionEvent row under natural-key idempotency.

        Returns the canonical persisted event per M7 (b) — the just-
        inserted event on insert path, or the existing row's event on
        natural-key no-op path. The producer (trace node) MUST use the
        returned event (not the incoming one) to construct the
        state-layer `TraceDecision` for the state delta, keeping state
        and audit in lockstep across retry/replay.
        """
        # Narrow the union return back to TraceDecisionEvent — registry
        # routes by event_type so the runtime instance matches.
        result = await self._persist_keyed_by_natural_key(event)
        if not isinstance(result, TraceDecisionEvent):
            raise AuditPersisterSchemaInvariantError(
                event_id=event.event_id,
                invariant="emit_trace_decision return-type mismatch",
            )
        return result

    async def emit_hitl_request(self, event: HITLRequestEvent) -> HITLRequestEvent:
        """Persist a HITLRequestEvent row under natural-key idempotency
        on `(review_id)` (single-shot per review per the HITL Non-goals).

        Returns the canonical persisted event — incoming on insert path,
        existing row's event on no-op match. The HITL node MUST use the
        returned event to construct the state-layer `HITLRequest` for
        the state delta, keeping state and audit in lockstep across the
        resume body re-run.
        """
        result = await self._persist_keyed_by_natural_key(event)
        if not isinstance(result, HITLRequestEvent):
            raise AuditPersisterSchemaInvariantError(
                event_id=event.event_id,
                invariant="emit_hitl_request return-type mismatch",
            )
        return result

    async def emit_hitl_decision(self, event: HITLDecisionEvent) -> HITLDecisionEvent:
        """Persist a HITLDecisionEvent row under natural-key idempotency
        on `(review_id)` with identity-subset check on
        `decisions_content_hash`.

        Identical-content concurrent submissions absorb cleanly (returned
        event is the existing persisted row with matching content hash);
        divergent-content submissions raise
        `AuditPersisterHITLDecisionNaturalKeyConflict`. The endpoint's
        failure wrapper catches the conflict and logs
        `hitl_resume_natural_key_conflict` at WARNING level (not INFO)
        with a diagnostic note distinguishing the concurrent-loser case
        from the window-(f) crash-retry case; canonical window-(f)
        recovery (advance lifecycle from the audit row) is owned by
        `sweep/hitl_expiry.py::reclaim_stuck_hitl_states`.
        """
        result = await self._persist_keyed_by_natural_key(event)
        if not isinstance(result, HITLDecisionEvent):
            raise AuditPersisterSchemaInvariantError(
                event_id=event.event_id,
                invariant="emit_hitl_decision return-type mismatch",
            )
        return result

    async def query_prior_publish_event(self, *, review_id: UUID) -> PublishEvent | None:
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
        most-recent row via `ORDER BY timestamp DESC, sequence_number DESC
        LIMIT 1`. The `sequence_number DESC` tie-breaker is load-bearing
        — `timestamp` alone is not a total order across concurrent emits.
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
                # `timestamp DESC` alone is not a total order — two
                # PublishEvent rows for the same review_id can share a
                # timestamp under concurrent emits, in which case the
                # returned row would be nondeterministic across runs and
                # the idempotency pre-flight could silently flap. The
                # append-only `sequence_number IDENTITY` column on
                # `audit_events` is the deterministic tie-breaker
                # (per `db/models/audit_events.py:57` —
                # `UNIQUE(review_id, sequence_number)`).
                .order_by(
                    AuditEventRow.timestamp.desc(),
                    AuditEventRow.sequence_number.desc(),
                )
                .limit(1)
            )
            payload = await session.scalar(stmt)
        if payload is None:
            return None
        return PublishEvent.model_validate(payload)

    async def query_review_llm_aggregates(
        self, *, review_id: UUID, is_eval: bool
    ) -> ReviewLLMAggregates:
        """Sum a review's LLM-call metrics (count + tokens + cost).

        Read-only; opens its own per-call `AsyncSession` (no transaction) and
        delegates to `aggregate_review_llm_metrics` — the single shared SUM the
        dashboard read-API also calls, so the synthesize-emitted audit row and the
        dashboard badge are computed by one aggregation path (FUP-093). `is_eval`
        is the review's own flag, scoping out divergent eval rows per FUP-130.
        """
        async with self._session_factory() as session:
            return await aggregate_review_llm_metrics(session, review_id=review_id, is_eval=is_eval)

    # See DECISIONS.md#027 — V1 per-review publish-side advisory lock
    # (try-lock with bounded backoff, serialize-then-observe). The
    # rejected-alternative rationale (plain blocking vs single-shot
    # try-lock) lives in the docstring + the DECISIONS entry.
    @asynccontextmanager
    async def acquire_publish_lock(
        self,
        *,
        review_id: UUID,
        max_wait_seconds: float = 120.0,
        initial_backoff_seconds: float = 0.05,
        max_backoff_seconds: float = 1.0,
    ) -> AsyncIterator[None]:
        """Per-review advisory lock — try-lock with bounded backoff retry.

        Implementation: loop runs `pg_try_advisory_xact_lock(<lock_id>)`
        in a fresh session, where `lock_id` is the first 8 bytes of
        `review_id.bytes` interpreted as a signed int8. On acquire,
        holds that session+transaction open for the duration of the
        `yield` (the caller's critical section); the lock auto-releases
        on commit at context exit. On NOT-acquired, the probe session+
        transaction is closed immediately (rollback releases the held
        connection back to the pool) and the loop sleeps with
        exponential backoff before retrying. After `max_wait_seconds`
        from the FIRST probe, raises
        `AuditPersisterPublishLockAcquisitionTimeoutError`.

        Concurrency model: serialized-first-then-observe. Two FastAPI
        background tasks racing on the same `review_id` (e.g., a
        human-issued `/decide` resume paired with
        `sweep/hitl_expiry.py::reclaim_stuck_hitl_states`'s graph-driven
        `Command(resume=...)`) serialize through this lock. The SECOND
        task, on acquiring the lock, re-reads `query_prior_publish_event`
        INSIDE the critical section — if the first task succeeded, the
        prior event is observed and the publish node's existing Step 4
        short-circuit emits `IDEMPOTENTLY_SKIPPED` (now AUTHENTIC: the
        first task's `PublishEvent` is the observed evidence). If the
        first task crashed before emitting `PublishEvent`, the prior
        event is `None` and the second task POSTs through Step 7 — the
        audit row's absence is the correct authority, not the lock
        release.

        Why try-lock + backoff (NOT plain blocking
        `pg_advisory_xact_lock`): a blocking variant holds a connection
        for the entire wait. With N same-review contenders, blocking
        would pin N connections from the pool simultaneously — the
        winner's emit_* calls (which open additional fresh sessions per
        the per-emit session discipline) could be starved by the
        waiters' held connections. Try-lock + backoff releases the
        probe session between attempts, so waiters only occupy a
        connection during the brief probe (~ms), not during the
        backoff sleep. Connection pressure under contention drops
        from N held + winner's K transient to ~1 held + occasional
        probes — the winner's emit_* path stays unblocked.

        Why NOT pure `pg_try_advisory_xact_lock` with single-shot
        loser-skip: a try-lock loser that returns immediately cannot
        observe whether the winner actually committed the POST.
        Loser-skip → emit `IDEMPOTENTLY_SKIPPED` even when the winner
        crashed between lock acquisition and POST → publish lost.
        Bounded retry puts the loser BEHIND the winner's transaction
        boundary on the SUCCESSFUL acquire path (winner has released,
        and on the next retry our acquire succeeds and we re-read the
        committed `PublishEvent`). False-skip class eliminated.

        The advisory-lock key is the first 8 bytes of `review_id.bytes`
        as a signed int8. UUIDs are 128-bit; the 64-bit slice gives
        near-zero collision probability at any realistic review volume.
        Disjoint by construction from `SWEEP_LOCK_ID=0x4F5554524452_0001`
        (uniform UUID distribution makes a collision against the fixed
        sweep id negligible).

        Timeout semantics: `max_wait_seconds=120` covers typical
        publish wall-clock (1-30s GitHub POST plus N-comment writes)
        with headroom. On timeout, raises
        `AuditPersisterPublishLockAcquisitionTimeoutError`; the publish
        node's outer try/except wrapping `enter_async_context` catches
        and emits `PublishAttemptEvent(outcome=failed,
        failure_class="...PublishLockAcquisitionTimeoutError")` before
        re-raising. Operator investigates via audit log + dashboard;
        the row stays at its pre-publish lifecycle state for retry.

        Backoff schedule: starts at `initial_backoff_seconds` (default
        50ms), doubles each iteration up to `max_backoff_seconds` (cap
        1s). Deterministic (no jitter for V1; V1.5 may add jitter if
        multi-tenant scheduling synchronization becomes an issue).
        Total maximum probe count is bounded above by
        `max_wait_seconds / initial_backoff_seconds` (worst case ~2400
        probes at default config, in practice far fewer due to
        exponential growth).
        """
        # Validate timing kwargs at call time. Zero or negative values
        # would degenerate the loop into a tight retry on `await
        # asyncio.sleep(0)` or `sleep(<negative>)` — same shape as the
        # `lifespan_sweep_loop.start_periodic_sweep` interval guard.
        # Validation is here (not at constructor time) because these
        # are per-call kwargs, not persister-level config.
        if max_wait_seconds <= 0:
            msg = f"acquire_publish_lock: max_wait_seconds must be > 0; got {max_wait_seconds}"
            raise ValueError(msg)
        if initial_backoff_seconds <= 0:
            msg = (
                f"acquire_publish_lock: initial_backoff_seconds must be > 0; "
                f"got {initial_backoff_seconds}"
            )
            raise ValueError(msg)
        if max_backoff_seconds < initial_backoff_seconds:
            msg = (
                f"acquire_publish_lock: max_backoff_seconds ({max_backoff_seconds}) "
                f"must be >= initial_backoff_seconds ({initial_backoff_seconds})"
            )
            raise ValueError(msg)

        # Derive a 64-bit lock_id directly from the UUID's first 8 bytes
        # interpreted as a signed int8 (Postgres advisory locks take
        # int8). `hashtext(...)` returns int4 (32-bit) and at ~65k
        # distinct UUIDs the birthday paradox gives ~50% collision
        # probability — distinct reviews would falsely serialize.
        # Using UUID bytes directly drops collision probability to
        # ~zero at any realistic review volume; the namespace stays
        # disjoint from `SWEEP_LOCK_ID=0x4F55545244520001` because
        # UUIDs have ~2^120 distinct values and a uniform-distribution
        # collision against the fixed sweep id is negligible.
        lock_id = int.from_bytes(review_id.bytes[:8], byteorder="big", signed=True)
        deadline = time.monotonic() + max_wait_seconds
        backoff = initial_backoff_seconds
        while True:
            async with self._session_factory() as session, session.begin():
                result = await session.execute(
                    sa_text("SELECT pg_try_advisory_xact_lock(:lock_id)"),
                    {"lock_id": lock_id},
                )
                if bool(result.scalar_one()):
                    # Acquired. Hold session+transaction open for the
                    # caller's critical section; commit (and lock
                    # release) happens at the outer `async with` exit
                    # below.
                    yield
                    return
            # Not acquired. Inner `async with` already exited — session
            # rolled back, connection back in pool. Sleep with backoff
            # outside session scope, then retry.
            now = time.monotonic()
            if now >= deadline:
                raise AuditPersisterPublishLockAcquisitionTimeoutError(
                    review_id=review_id,
                    waited_seconds=int(max_wait_seconds),
                )
            sleep_for = min(backoff, deadline - now)
            await asyncio.sleep(sleep_for)
            backoff = min(backoff * 2, max_backoff_seconds)

    async def query_hitl_decision_event(self, *, review_id: UUID) -> HITLDecisionEvent | None:
        """Return the persisted `HITLDecisionEvent` for `review_id`, or None.

        Sister of `query_prior_publish_event`. The HITL audit row's
        natural-key partial unique index (one row per review_id per
        spec Group 3) means there's AT MOST ONE event per review;
        the ORDER BY is defensive — if the index is somehow violated
        (e.g., migration regression), the most-recent row wins by
        `(timestamp, sequence_number) DESC` so consumers see the
        canonical "what decision actually landed".

        Used by `sweep/hitl_expiry.py::reclaim_stuck_hitl_states` for
        window-(f) crash recovery: when `emit_hitl_decision` succeeded
        but `mark_running` never landed, the audit row IS the canonical
        decision. The sweep reads this row, reconstructs the canonical
        `HITLDecision` from its fields, and drives the graph through
        `Command(resume=canonical_decision.model_dump(mode="json"))` —
        the body re-runs the hitl node from the top, the natural-key
        check returns the existing event, `mark_running` writes the
        canonical JSONB inside the body, phase end emits, the graph
        routes to publish, and publish runs against the recovered
        finding set. A direct `mark_running` write from the sweep
        would advance the lifecycle column but leave the graph
        permanently suspended at the HITL interrupt — `/decide` would
        409-reject all retries (preflight sees `hitl_decision IS NOT
        NULL`) and the gated findings never reach GitHub.

        Read-only — opens its own `AsyncSession` (no `session.begin()`,
        no transaction needed for a single SELECT). Per-call session
        discipline mirrors `query_prior_publish_event`.
        """
        hitl_decision_event_type: str = HITLDecisionEvent.model_fields["event_type"].default
        async with self._session_factory() as session:
            stmt = (
                select(AuditEventRow.payload)
                .where(
                    AuditEventRow.review_id == review_id,
                    AuditEventRow.event_type == hitl_decision_event_type,
                )
                .order_by(
                    AuditEventRow.timestamp.desc(),
                    AuditEventRow.sequence_number.desc(),
                )
                .limit(1)
            )
            payload = await session.scalar(stmt)
        if payload is None:
            return None
        return HITLDecisionEvent.model_validate(payload)

    async def query_replay_verdict_event(self, *, review_id: UUID) -> ReplayVerdictEvent | None:
        """Return the persisted `ReplayVerdictEvent` for `review_id`, or None.

        Sister of `query_hitl_decision_event`. The partial unique index
        `uq_audit_events_replay_verdict_natural_key` means AT MOST ONE verdict per
        review; the ORDER BY is defensive (most-recent wins if the index is somehow
        violated). Read-only — its own `AsyncSession`, no transaction.
        """
        verdict_event_type: str = ReplayVerdictEvent.model_fields["event_type"].default
        async with self._session_factory() as session:
            stmt = (
                select(AuditEventRow.payload)
                .where(
                    AuditEventRow.review_id == review_id,
                    AuditEventRow.event_type == verdict_event_type,
                )
                .order_by(
                    AuditEventRow.timestamp.desc(),
                    AuditEventRow.sequence_number.desc(),
                )
                .limit(1)
            )
            payload = await session.scalar(stmt)
        if payload is None:
            return None
        return ReplayVerdictEvent.model_validate(payload)
