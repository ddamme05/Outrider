"""Outrider ORM models — public re-exports.

Import order matters for SQLAlchemy's metadata: parent tables before tables
that FK into them. Within independent tables (severity_policies, installations),
order is alphabetical for stability.

The `Base` and the two PG ENUMs live in `_base.py` to keep `__init__.py` free
of declarative state and avoid any circular-import risk via the model files'
`from outrider.db.models._base import Base` pattern.
"""

from outrider.db.models._base import Base, anomaly_status_enum, review_status_enum
from outrider.db.models.analyze_file_cache import AnalyzeFileCache
from outrider.db.models.anomalies import Anomaly
from outrider.db.models.audit_events import AuditEvent
from outrider.db.models.findings import Finding
from outrider.db.models.installations import Installation, InstallationRepository
from outrider.db.models.llm_call_content import LLMCallContent
from outrider.db.models.purge_audit import PurgeAudit
from outrider.db.models.reviews import Review
from outrider.db.models.severity_policies import SeverityPolicy

__all__ = [
    "AnalyzeFileCache",
    "Anomaly",
    "AuditEvent",
    "Base",
    "Finding",
    "Installation",
    "InstallationRepository",
    "LLMCallContent",
    "PurgeAudit",
    "Review",
    "SeverityPolicy",
    "anomaly_status_enum",
    "review_status_enum",
]
