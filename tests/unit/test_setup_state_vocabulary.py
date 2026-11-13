"""Check setup status values against ORM and migration constraints.

The migration is a fixed snapshot; vocabulary changes must update it too.
See DECISIONS.md#070."""

from __future__ import annotations

from pathlib import Path

from outrider.db.models.setup_state import _STATUS_CHECK, SETUP_STATUSES

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "db"
    / "migrations"
    / "versions"
    / "b3f9a1c72e04_app_manifest_onboarding_tables.py"
)


def test_status_vocabulary_is_the_five_states() -> None:
    assert SETUP_STATUSES == (
        "UNCONFIGURED",
        "AWAITING_CALLBACK",
        "CONVERTING",
        "CONFIGURED",
        "ORPHANED",
    )


def test_model_check_covers_every_status() -> None:
    for status in SETUP_STATUSES:
        assert f"'{status}'" in _STATUS_CHECK


def test_migration_check_matches_vocabulary() -> None:
    """The onboarding migration's hardcoded `status IN (...)` must list exactly SETUP_STATUSES."""
    text = _MIGRATION.read_text()
    for status in SETUP_STATUSES:
        assert f"'{status}'" in text, f"{status} missing from the migration CHECK"
    # No stray status in the migration that isn't in the vocabulary.
    import re

    m = re.search(r"status IN \(([^)]*)\)", text)
    assert m is not None
    listed = {s.strip().strip("'") for s in m.group(1).split(",")}
    assert listed == set(SETUP_STATUSES)
