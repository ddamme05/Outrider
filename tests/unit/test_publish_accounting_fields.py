"""PublishResult + PublishEvent three-channel accounting fields (DECISIONS.md#050).

Commit 1 of publish-review-body-materialization: the two new count fields
(`review_body_findings_posted`, `dashboard_only_findings_surfaced`) exist on both
the state result and the audit event, default to 0 (so the existing factories +
the current emit stay valid — the routing loop threads the real counts in a later
commit), and a historical `PublishEvent` row lacking them reconstructs cleanly
under `AuditEventAdapter` (replay historical-tolerance). `comments_posted` stays
the inline count.
"""

from __future__ import annotations

from uuid import uuid4

from outrider.audit.events import AuditEventAdapter, PublishEvent
from outrider.schemas.publish import PublishResult


def test_publish_result_factories_default_new_channels_to_zero() -> None:
    """All four factories produce review_body/dashboard counts = 0 (additive change
    does not break them; the real counts arrive with the routing-loop commit)."""
    for result in (
        PublishResult.success(github_review_id=7, comments_posted=3),
        PublishResult.empty(),
        PublishResult.skipped(),
        PublishResult.skipped_external(existing_review_id=9),
    ):
        assert result.review_body_findings_posted == 0
        assert result.dashboard_only_findings_surfaced == 0


def test_publish_result_accepts_explicit_channels() -> None:
    """The three channels round-trip independently (forward-compat for the routing
    loop that will set them)."""
    result = PublishResult(
        outcome="success",
        github_review_id=7,
        comments_posted=3,
        review_body_findings_posted=2,
        dashboard_only_findings_surfaced=1,
    )
    assert result.comments_posted == 3
    assert result.review_body_findings_posted == 2
    assert result.dashboard_only_findings_surfaced == 1


def test_publish_event_defaults_new_channels_to_zero() -> None:
    """The current emit (which sets only comments_posted) yields 0 for both new
    channels."""
    event = PublishEvent(
        review_id=uuid4(), github_review_id=1, comments_posted=2, review_status="COMMENT"
    )
    assert event.review_body_findings_posted == 0
    assert event.dashboard_only_findings_surfaced == 0


def test_publish_event_historical_row_reconstructs_without_new_fields() -> None:
    """A pre-#050 PublishEvent payload (lacking the two new fields) reconstructs via
    AuditEventAdapter with them defaulted to 0 — replay historical-tolerance, so
    older audit rows do not 500 the reconstructor."""
    event = PublishEvent(
        review_id=uuid4(), github_review_id=1, comments_posted=2, review_status="COMMENT"
    )
    payload = event.model_dump(mode="json")
    del payload["review_body_findings_posted"]
    del payload["dashboard_only_findings_surfaced"]

    reconstructed = AuditEventAdapter.validate_python(payload)
    assert isinstance(reconstructed, PublishEvent)
    assert reconstructed.review_body_findings_posted == 0
    assert reconstructed.dashboard_only_findings_surfaced == 0
    assert reconstructed.comments_posted == 2
