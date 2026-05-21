"""Audit event hierarchy + discriminated union per docs/spec.md §7.2.1 + §8.2.

Re-exports the public symbols of the audit package. The emitter
(`audit/emitter.py`, separate spec) constructs concrete events; replay
(`audit/replay.py`, separate spec) reconstructs them via `AuditEvent`.
The durable persister (`audit/persister.py`) implements four Protocol
contracts atomically per `DECISIONS.md#016`: `LLMExchangePersister`
(`llm/base.py`) plus `PhaseEventSink`, `FileExaminationSink`, and
`AnalyzeEventSink` (all in `audit/sinks.py`).
"""

from outrider.audit.config import RetentionSettings
from outrider.audit.events import (
    AgentTransitionEvent,
    AuditEvent,
    AuditEventAdapter,
    AuditEventBase,
    ContextManifestEntry,
    FileExaminationEvent,
    FindingEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    LLMCallEvent,
    PublishEvent,
    PublishRoutingEvent,
    ReviewPhaseEvent,
    TraceDecisionEvent,
    compute_finding_content_hash,
)
from outrider.audit.persister import (
    METADATA_ONLY_EXCEPTION_TYPES,
    AuditPersister,
    AuditPersisterConfigError,
    AuditPersisterEventRequestFieldMismatchError,
    AuditPersisterEventResponseFieldMismatchError,
    AuditPersisterIdempotencyConflict,
    AuditPersisterReviewIdMismatchError,
    AuditPersisterReviewNotFoundError,
    AuditPersisterSchemaInvariantError,
    FieldDigest,
)
from outrider.audit.sinks import PhaseEventSink

__all__ = [
    "AgentTransitionEvent",
    "AuditEvent",
    "AuditEventAdapter",
    "AuditEventBase",
    "AuditPersister",
    "AuditPersisterConfigError",
    "AuditPersisterEventRequestFieldMismatchError",
    "AuditPersisterEventResponseFieldMismatchError",
    "AuditPersisterIdempotencyConflict",
    "AuditPersisterReviewIdMismatchError",
    "AuditPersisterReviewNotFoundError",
    "AuditPersisterSchemaInvariantError",
    "ContextManifestEntry",
    "FieldDigest",
    "METADATA_ONLY_EXCEPTION_TYPES",
    "FileExaminationEvent",
    "FindingEvent",
    "HITLDecisionEvent",
    "HITLRequestEvent",
    "LLMCallEvent",
    "PhaseEventSink",
    "PublishEvent",
    "PublishRoutingEvent",
    "RetentionSettings",
    "ReviewPhaseEvent",
    "TraceDecisionEvent",
    "compute_finding_content_hash",
]
