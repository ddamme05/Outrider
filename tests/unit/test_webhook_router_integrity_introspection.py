"""Check duplicate-review constraint classification with fabricated database errors.

The predicate returns True only for uq_review_natural_key. False sends the
handler to a natural-key lookup, which can still return a duplicate response.
Real transaction and handler outcomes live in
tests/integration/test_webhook_router_integration.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sqlalchemy.exc import IntegrityError as SQLAlchemyIntegrityError

from outrider.api.webhooks.router import _is_reviews_natural_key_conflict


def _fabricate_integrity_error(
    *,
    orig: Any = None,
) -> SQLAlchemyIntegrityError:
    """Construct a `SQLAlchemyIntegrityError` whose `.orig` is the supplied
    object (or None). `statement` and `params` arguments are required by
    the constructor but irrelevant to the introspection logic."""
    err = SQLAlchemyIntegrityError(
        statement="INSERT INTO reviews ...",
        params={},
        orig=orig,
    )
    # SQLAlchemy's __init__ wraps `orig` — re-attach for predictable access
    # in case the wrapping path replaces it with something opaque.
    err.orig = orig
    return err


# ---------------------------------------------------------------------------
# True cases — natural-key conflict
# ---------------------------------------------------------------------------


def test_uq_review_natural_key_constraint_name_returns_true() -> None:
    """The reviews natural-key constraint returns True from the classifier."""
    orig = SimpleNamespace(
        diag=SimpleNamespace(constraint_name="uq_review_natural_key"),
    )
    err = _fabricate_integrity_error(orig=orig)
    assert _is_reviews_natural_key_conflict(err) is True


# ---------------------------------------------------------------------------
# False cases, not a reviews natural-key conflict
# ---------------------------------------------------------------------------


def test_audit_events_pkey_constraint_returns_false() -> None:
    """An audit primary-key collision is not a reviews natural-key conflict."""
    orig = SimpleNamespace(
        diag=SimpleNamespace(constraint_name="audit_events_pkey"),
    )
    err = _fabricate_integrity_error(orig=orig)
    assert _is_reviews_natural_key_conflict(err) is False


def test_arbitrary_other_constraint_returns_false() -> None:
    """Any constraint name other than `uq_review_natural_key` →
    False. Defends against future constraints (e.g., FK violations) being
    misclassified."""
    orig = SimpleNamespace(
        diag=SimpleNamespace(constraint_name="reviews_installation_id_fkey"),
    )
    err = _fabricate_integrity_error(orig=orig)
    assert _is_reviews_natural_key_conflict(err) is False


def test_constraint_name_none_returns_false() -> None:
    """A missing constraint name does not establish a reviews natural-key conflict."""
    orig = SimpleNamespace(
        diag=SimpleNamespace(constraint_name=None),
    )
    err = _fabricate_integrity_error(orig=orig)
    assert _is_reviews_natural_key_conflict(err) is False


def test_missing_diag_attribute_returns_false() -> None:
    """`exc.orig.diag` is absent entirely (driver doesn't expose it) →
    False. Forward-compat for drivers other than psycopg3."""
    # An object with NO `diag` attribute at all.
    orig = SimpleNamespace()
    err = _fabricate_integrity_error(orig=orig)
    assert _is_reviews_natural_key_conflict(err) is False


def test_exc_orig_is_none_returns_false() -> None:
    """`exc.orig` is None (no underlying driver error attached) → False.
    Edge case: an `IntegrityError` constructed without an underlying
    driver exception."""
    err = _fabricate_integrity_error(orig=None)
    assert _is_reviews_natural_key_conflict(err) is False
