"""Declarative base + the two PostgreSQL ENUM types declared at MetaData scope.

Living separately from `__init__.py` so model files can `from ._base import Base`
without triggering a circular import through the package init's re-exports.

The two PG ENUMs are intentionally distinct database objects per docs/schema.md
"Two distinct PG ENUM types" — sharing one would block adding a value to one
status column without affecting the other and would cross-pollute static checking.
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
