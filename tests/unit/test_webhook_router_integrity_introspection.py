"""Unit tests for `_is_reviews_natural_key_conflict` — the constraint-name
introspection that distinguishes "duplicate webhook delivery" from
"audit-side IntegrityError that must re-raise."

The end-to-end IntegrityError-race scenarios are hard to orchestrate
deterministically (the conflict requires another transaction to commit
the colliding row BETWEEN the fast-path SELECT and the slow-path
INSERT — a real race). These unit tests cover the load-bearing
introspection function directly: fabricate `SQLAlchemyIntegrityError`
objects with the various `exc.orig.diag.constraint_name` values the
function decides against, assert correct classification.

Cases pinned:
  1. constraint_name == "uq_review_natural_key" → True (duplicate
     delivery; the router short-circuits to existing review_id with 200).
  2. constraint_name == "audit_events_pkey" → False (audit-side PK
     collision; the router re-raises so GitHub retries).
  3. constraint_name == None / missing → False (driver doesn't expose
     diag; safer to re-raise than guess).
  4. exc.orig is None → False (no underlying driver error; re-raise).

The combined coverage of these four cases + the existing integration
tests for fast-path 200 / unknown-installation 4xx / membership 4xx /
happy-path 202 exercises every branch of the natural-key conflict
discriminator the spec mandates.
"""

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
    """psycopg3 diag.constraint_name == 'uq_review_natural_key' →
    classified as duplicate delivery. The router short-circuits to
    200 with existing review_id."""
    orig = SimpleNamespace(
        diag=SimpleNamespace(constraint_name="uq_review_natural_key"),
    )
    err = _fabricate_integrity_error(orig=orig)
    assert _is_reviews_natural_key_conflict(err) is True


# ---------------------------------------------------------------------------
# False cases — re-raise required
# ---------------------------------------------------------------------------


def test_audit_events_pkey_constraint_returns_false() -> None:
    """An IntegrityError from the audit_events PK collision (different
    constraint name) MUST re-raise — misclassifying as duplicate would
    silently swallow audit-side conflicts and lose the review-creation
    work. Spec's load-bearing distinction."""
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
    """`diag.constraint_name` is None (driver couldn't determine the
    constraint) → fail-loud re-raise. Better than guessing duplicate
    when uncertain."""
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
