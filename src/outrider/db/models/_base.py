"""Share the SQLAlchemy model base and database enum definitions.

Review and anomaly statuses use separate database enum types even where their
values overlap. Keeping the shared declarations here avoids model-import
cycles.
"""

from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


review_status_enum = ENUM(
    "running",
    "awaiting_approval",
    "awaiting_approval_expired",
    "completed",
    "failed",
    "skipped",
    name="review_status_enum",
    metadata=Base.metadata,
)


anomaly_status_enum = ENUM(
    "open",
    "acknowledged",
    "resolved",
    name="anomaly_status_enum",
    metadata=Base.metadata,
)
