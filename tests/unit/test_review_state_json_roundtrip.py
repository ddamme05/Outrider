"""`ReviewState` JSON round-trip — pins `state-is-pure-data`.

The V1 `BackgroundTasksDispatcher.dispatch` calls `state.model_dump_json()`
(discarded) on every dispatch as a fail-loud gate for any future
contributor who adds a non-JSON-serializable field to `ReviewState` or
its nested models. This test exercises the same gate at the schema
level: a seed state model-dumps-as-JSON cleanly AND round-trips back to
an equal `ReviewState` via `model_validate_json`.

If a future schema extension breaks JSON round-trip, this test fails
BEFORE the dispatcher's gate fires — V1 isn't a free pass for V2's
Celery serialization.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from outrider.agent.state import ReviewState
from outrider.schemas.pr_context import ChangedFile, PRContext


def _build_seed() -> ReviewState:
    """A seed state matching the webhook-receipt shape per DECISIONS.md#020."""
    pr_context = PRContext(
        installation_id=12345,
        owner="acme",
        repo="widgets",
        pr_number=42,
        pr_title="Test PR",
        pr_body="Body text",
        base_sha="b" * 40,
        head_sha="h" * 40,
        author="alice",
        total_additions=5,
        total_deletions=2,
        changed_files=(),
    )
    return ReviewState(
        review_id=uuid4(),
        pr_context=pr_context,
        received_at=datetime.now(UTC),
        is_eval=False,
    )


def _build_enriched_state() -> ReviewState:
    """Post-intake state with a populated ChangedFile tuple.

    The dispatcher's gate fires at receipt time when changed_files is
    empty, but the same field shapes flow back through state on
    LangGraph checkpoint replay; round-trip must hold for enriched
    state too.
    """
    pr_context = PRContext(
        installation_id=12345,
        owner="acme",
        repo="widgets",
        pr_number=42,
        pr_title="Test PR",
        pr_body=None,
        base_sha="b" * 40,
        head_sha="h" * 40,
        author="alice",
        total_additions=5,
        total_deletions=2,
        changed_files=(
            ChangedFile(
                path="src/example.py",
                status="modified",
                additions=5,
                deletions=2,
                patch="@@ -1 +1 @@\n-old\n+new\n",
                content_base="old\n",
                content_head="new\n",
                previous_path=None,
            ),
        ),
    )
    return ReviewState(
        review_id=uuid4(),
        pr_context=pr_context,
        received_at=datetime.now(UTC),
        is_eval=False,
    )


def test_seed_state_roundtrips_through_json() -> None:
    """Webhook-seed shape (empty changed_files) round-trips cleanly."""
    seed = _build_seed()
    serialized = seed.model_dump_json()
    rehydrated = ReviewState.model_validate_json(serialized)

    assert rehydrated.review_id == seed.review_id
    assert rehydrated.received_at == seed.received_at
    assert rehydrated.is_eval == seed.is_eval
    assert rehydrated.pr_context.installation_id == seed.pr_context.installation_id
    assert rehydrated.pr_context.changed_files == ()


def test_post_intake_state_roundtrips_through_json() -> None:
    """Post-intake shape (populated changed_files) round-trips cleanly."""
    state = _build_enriched_state()
    serialized = state.model_dump_json()
    rehydrated = ReviewState.model_validate_json(serialized)

    assert rehydrated.review_id == state.review_id
    assert len(rehydrated.pr_context.changed_files) == 1
    rcf = rehydrated.pr_context.changed_files[0]
    ocf = state.pr_context.changed_files[0]
    assert rcf.path == ocf.path
    assert rcf.status == ocf.status
    assert rcf.content_base == ocf.content_base
    assert rcf.content_head == ocf.content_head
    assert rcf.previous_path == ocf.previous_path


def test_is_eval_true_roundtrips() -> None:
    """The eval-isolation flag survives JSON round-trip."""
    pr_context = PRContext(
        installation_id=12345,
        owner="acme",
        repo="widgets",
        pr_number=42,
        pr_title="Eval PR",
        pr_body=None,
        base_sha="b" * 40,
        head_sha="h" * 40,
        author="eval-fixture",
        total_additions=0,
        total_deletions=0,
        changed_files=(),
    )
    state = ReviewState(
        review_id=uuid4(),
        pr_context=pr_context,
        received_at=datetime.now(UTC),
        is_eval=True,
    )
    rehydrated = ReviewState.model_validate_json(state.model_dump_json())
    assert rehydrated.is_eval is True


def test_aware_datetime_survives_roundtrip() -> None:
    """`AwareDatetime` rehydrates with the same UTC tz."""
    state = _build_seed()
    rehydrated = ReviewState.model_validate_json(state.model_dump_json())
    assert rehydrated.received_at.tzinfo is not None
    # The UTC offset is preserved.
    assert rehydrated.received_at.utcoffset() == state.received_at.utcoffset()
