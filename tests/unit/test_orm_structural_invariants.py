"""ORM-metadata structural invariants enforced before alembic generates a migration.

Three Python-level guards on the declarative metadata. Each one mirrors
an integration-tier test that asserts the same property at the live-DB
introspection level — the unit-tier check fails fast (no DB needed) when
a developer changes a model in a way that violates the invariant, and
the integration-tier check catches anything that slips past metadata
(hand-edited migrations, autogenerate quirks).

  - All ``_at`` (and ``timestamp``) columns use ``TIMESTAMP(timezone=True)``.
    Mirrors ``test_timestamp_columns_aware``.
  - The ``Finding`` model declares no ``confidence`` column.
    Per ``confidence-is-computed-not-assigned``: confidence is a
    computed_field on the Pydantic model, derived from evidence_tier
    deterministically. The DB column would be a tempting place to
    persist it, but the architecture deliberately does not.
  - ``AuditEvent.review_id`` is plain ``Mapped[UUID]`` with no
    ``ForeignKey``. Mirrors
    ``test_audit_events_has_no_foreign_key_constraints``.

Pure unit tests — no DB connection, no fixtures.
"""

from sqlalchemy import DateTime

from outrider.db.models import AuditEvent, Base, Finding


def test_all_at_columns_are_timezone_aware() -> None:
    """Every column ending in _at or named 'timestamp' is TIMESTAMP(timezone=True)."""
    violations = []
    for table in Base.metadata.tables.values():
        for col in table.columns:
            if not (col.name.endswith("_at") or col.name == "timestamp"):
                continue
            if not isinstance(col.type, DateTime):
                violations.append((table.name, col.name, type(col.type).__name__, "not DateTime"))
            elif not col.type.timezone:
                violations.append((table.name, col.name, "DateTime", "timezone=False"))

    assert violations == [], (
        f"Found timestamp columns NOT declared with timezone=True: {violations}"
    )


def test_finding_model_has_no_confidence_column() -> None:
    """Finding must not declare a `confidence` column.

    Per `confidence-is-computed-not-assigned`: confidence is a
    @computed_field on the Pydantic model, derived from evidence_tier
    deterministically. Persisting it in the DB would let a future bug
    write a model-set value into a column that's supposed to be derived,
    silently bypassing the rule.
    """
    finding_columns = {col.name for col in Finding.__table__.columns}
    assert "confidence" not in finding_columns, (
        f"Finding declares a `confidence` column: {finding_columns}. "
        "Confidence is computed from evidence_tier, not stored. See "
        "`confidence-is-computed-not-assigned`."
    )


def test_audit_event_review_id_has_no_foreign_key() -> None:
    """AuditEvent.review_id is plain Mapped[UUID] without ForeignKey.

    Per docs/schema.md "AUDIT_EVENTS.review_id is a logical reference,
    not a DB FK": no cascade behavior fits both append-only-forever and
    the always-non-null invariant. Adding a ForeignKey to the column
    would re-introduce one of the rejected cascade behaviors and break
    the metadata-only-replay state per #014 point 4.
    """
    review_id_col = AuditEvent.__table__.columns["review_id"]
    fks = list(review_id_col.foreign_keys)
    assert fks == [], f"AuditEvent.review_id must declare no ForeignKey; found: {fks}"
