"""Audit event hierarchy + discriminated union per docs/spec.md §7.2.1 + §8.2.

Re-exports the public symbols of the audit package. The emitter
(`audit/emitter.py`, separate spec) constructs concrete events; replay
(`audit/replay.py`) reconstructs them via `AuditEvent` and exposes
`AuditReplayer` (re-exported below).
The durable persister (`audit/persister.py`) implements four Protocol
contracts atomically per `DECISIONS.md#016`: `LLMExchangePersister`
(`llm/base.py`) plus `PhaseEventSink`, `FileExaminationSink`, and
`AnalyzeEventSink` (all in `audit/sinks.py`).

Also re-exports `RetentionSettings` from `audit/config.py` for callers
that wire retention-aware persisters at lifespan-construction time.
"""

from outrider.audit.config import RetentionSettings
from outrider.audit.events import (
    AgentTransitionEvent,
    AnalyzeCompletedEvent,
    AnalyzeResponseRejectedEvent,
    AuditEvent,
    AuditEventAdapter,
    AuditEventBase,
    ContextManifestEntry,
    FileExaminationEvent,
    FindingEvent,
    FindingProposalRejectedEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    LLMCallEvent,
    PublishEvent,
    PublishRoutingEvent,
    ReviewPhaseEvent,
    SynthesizeCompletedEvent,
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
from outrider.audit.replay import (
    AuditReplayer,
    FindingContent,
    ReconstructedFinding,
    ReconstructedLLMExchange,
    ReconstructedPhase,
    ReconstructedReview,
    ReconstructedReviewMetadata,
    ReplayEquivalenceError,
    ReplayError,
    ReplayMode,
    ReplayReviewNotFoundError,
)
from outrider.audit.sinks import AnalyzeEventSink, FileExaminationSink, PhaseEventSink

__all__ = [
    "AgentTransitionEvent",
    "AnalyzeCompletedEvent",
    "AnalyzeEventSink",
    "AnalyzeResponseRejectedEvent",
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
    "AuditReplayer",
    "ContextManifestEntry",
    "FieldDigest",
    "METADATA_ONLY_EXCEPTION_TYPES",
    "FileExaminationEvent",
    "FileExaminationSink",
    "FindingContent",
    "FindingEvent",
    "FindingProposalRejectedEvent",
    "HITLDecisionEvent",
    "HITLRequestEvent",
    "LLMCallEvent",
    "PhaseEventSink",
    "PublishEvent",
    "PublishRoutingEvent",
    "ReconstructedFinding",
    "ReconstructedLLMExchange",
    "ReconstructedPhase",
    "ReconstructedReview",
    "ReconstructedReviewMetadata",
    "ReplayEquivalenceError",
    "ReplayError",
    "ReplayMode",
    "ReplayReviewNotFoundError",
    "RetentionSettings",
    "ReviewPhaseEvent",
    "SynthesizeCompletedEvent",
    "TraceDecisionEvent",
    "compute_finding_content_hash",
]
