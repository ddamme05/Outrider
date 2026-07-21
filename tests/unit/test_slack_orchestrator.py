"""SlackNotificationOrchestrator — dedup, mutual exclusion, post-then-record, fire-and-forget.

Pins the two V1 flows (`notify_hitl_pending` / `notify_review_posted`): a clean
post records a `SlackNotificationEvent` carrying the returned `ts`; the pre-post
dedup + mutual-exclusion guards skip re-posts; and every failure path (transport
error, audit-emit crash, dedup-query crash) is swallowed so the graph never sees
an exception. The notifier + sink are in-memory doubles. See
specs/2026-06-15-slack-dashboard-in-slack.md (Output boundary / Audit append-only).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from outrider.audit.events import SlackNotificationEvent, compute_finding_content_hash
from outrider.notify.base import SlackChannelError, SlackPostResult
from outrider.notify.orchestrator import SlackNotificationOrchestrator
from outrider.policy import EvidenceTier
from outrider.policy.severity import ACTIVE_POLICY_VERSION, FindingSeverity, FindingType
from outrider.schemas import ReviewDimension
from outrider.schemas.review_finding import ReviewFinding

if TYPE_CHECKING:
    from outrider.notify.base import SlackBlocks

_BASE_URL = "https://dash.example.com"
_KIND = Literal["hitl_pending", "review_posted"]


class _FakeNotifier:
    """`SlackNotifier` double: records posts/updates, returns a `SlackPostResult`,
    optionally raises a transport error instead of posting."""

    def __init__(self, *, ts: str = "1700000000.000100", raise_on_post: Exception | None = None):
        self.posts: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self._ts = ts
        self._raise_on_post = raise_on_post

    async def post_message(
        self, *, channel: str, text: str, blocks: SlackBlocks | None = None
    ) -> SlackPostResult:
        self.posts.append({"channel": channel, "text": text, "blocks": blocks})
        if self._raise_on_post is not None:
            raise self._raise_on_post
        return SlackPostResult(channel=channel, ts=self._ts)

    async def update_message(
        self, *, channel: str, ts: str, text: str, blocks: SlackBlocks | None = None
    ) -> None:
        self.updates.append({"channel": channel, "ts": ts, "text": text})

    async def aclose(self) -> None:
        return None


class _FakeSink:
    """`SlackEventSink` double: records emits; `query_slack_notification` reflects
    both pre-seeded rows and prior emits (most-recent match), so dedup behaves
    like the real natural-key lookup. Can raise on emit or query."""

    def __init__(
        self,
        *,
        existing: dict[tuple[UUID, str, str], SlackNotificationEvent] | None = None,
        raise_on_emit: Exception | None = None,
        raise_on_query: Exception | None = None,
        fail_first_emit: bool = False,
    ):
        self.emitted: list[SlackNotificationEvent] = []
        self._existing = existing or {}
        self._raise_on_emit = raise_on_emit
        self._raise_on_query = raise_on_query
        self._fail_first_emit = fail_first_emit
        self._emit_calls = 0

    async def emit_slack_notification(self, event: SlackNotificationEvent) -> None:
        self._emit_calls += 1
        if self._raise_on_emit is not None:
            raise self._raise_on_emit
        if self._fail_first_emit and self._emit_calls == 1:
            raise RuntimeError("audit insert failed (first attempt)")
        self.emitted.append(event)

    async def query_slack_notification(
        self, *, review_id: UUID, channel_id: str, kind: str
    ) -> SlackNotificationEvent | None:
        if self._raise_on_query is not None:
            raise self._raise_on_query
        seeded = self._existing.get((review_id, channel_id, kind))
        if seeded is not None:
            return seeded
        for event in reversed(self.emitted):
            if (
                event.review_id == review_id
                and event.channel_id == channel_id
                and event.kind == kind
            ):
                return event
        return None


def _orch(notifier: _FakeNotifier, sink: _FakeSink) -> SlackNotificationOrchestrator:
    return SlackNotificationOrchestrator(notifier=notifier, sink=sink, dashboard_base_url=_BASE_URL)


def _slack_event(review_id: UUID, channel_id: str, kind: _KIND) -> SlackNotificationEvent:
    return SlackNotificationEvent(
        review_id=review_id,
        is_eval=False,
        channel_id=channel_id,
        message_ts="1699999999.000001",
        kind=kind,
        posted_at=datetime(2026, 6, 15, tzinfo=UTC),
    )


def _finding(severity: FindingSeverity, *, title: str = "f") -> ReviewFinding:
    ft = {
        FindingSeverity.CRITICAL: FindingType.SQL_INJECTION,
        FindingSeverity.HIGH: FindingType.HARDCODED_SECRET,
    }[severity]
    fp, line = "src/x.py", 10
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=1,
        finding_type=ft,
        dimension=ReviewDimension.SECURITY,
        severity=severity,
        file_path=fp,
        line_start=line,
        line_end=line,
        title=title,
        description="DESCRIPTION MUST NOT REACH SLACK",
        evidence="EVIDENCE MUST NOT REACH SLACK",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=fp, line_start=line, line_end=line, finding_type=ft
        ),
        proposal_hash=hashlib.sha256(f"{ft}{line}{title}".encode()).hexdigest(),
    )


async def test_notify_hitl_pending_posts_and_records() -> None:
    """Clean post: one message to the channel + one hitl_pending event carrying the
    returned ts and the deep-link; metadata-first (no description/evidence leaks)."""
    notifier, sink = _FakeNotifier(ts="123.456"), _FakeSink()
    rid = uuid4()
    result = await _orch(notifier, sink).notify_hitl_pending(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="Add login",
        findings=[_finding(FindingSeverity.CRITICAL, title="sqli")],
    )
    assert result is not None
    assert result.ts == "123.456"

    assert len(notifier.posts) == 1
    post = notifier.posts[0]
    assert post["channel"] == "C1"
    assert post["blocks"] is not None
    assert f"{_BASE_URL}/reviews/{rid}" in post["text"]
    assert "DESCRIPTION MUST NOT REACH SLACK" not in post["text"]

    assert len(sink.emitted) == 1
    event = sink.emitted[0]
    assert isinstance(event, SlackNotificationEvent)
    assert event.kind == "hitl_pending"
    assert event.message_ts == "123.456"
    assert event.channel_id == "C1"
    assert event.review_id == rid
    assert event.is_eval is False


async def test_notify_hitl_pending_dedup_skips_when_already_posted() -> None:
    """A pre-existing hitl_pending row (the replay case) suppresses the re-post."""
    rid = uuid4()
    notifier = _FakeNotifier()
    sink = _FakeSink(
        existing={(rid, "C1", "hitl_pending"): _slack_event(rid, "C1", "hitl_pending")}
    )
    result = await _orch(notifier, sink).notify_hitl_pending(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="t",
        findings=[],
    )
    assert result is None
    assert notifier.posts == []
    assert sink.emitted == []


async def test_notify_hitl_pending_idempotent_across_two_calls() -> None:
    """Calling twice with the same key posts + records once; the second call dedups
    against the first call's emitted row."""
    notifier, sink = _FakeNotifier(), _FakeSink()
    orch, rid = _orch(notifier, sink), uuid4()
    findings = [_finding(FindingSeverity.HIGH)]
    first = await orch.notify_hitl_pending(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="t",
        findings=findings,
    )
    second = await orch.notify_hitl_pending(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="t",
        findings=findings,
    )
    assert first is not None
    assert second is None
    assert len(notifier.posts) == 1
    assert len(sink.emitted) == 1


async def test_notify_hitl_pending_swallows_post_failure() -> None:
    """A transport failure degrades to None with no audit row + no exception."""
    notifier = _FakeNotifier(raise_on_post=SlackChannelError("not_in_channel"))
    sink = _FakeSink()
    result = await _orch(notifier, sink).notify_hitl_pending(
        review_id=uuid4(),
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="t",
        findings=[],
    )
    assert result is None
    assert len(notifier.posts) == 1  # the post was attempted; the transport raised
    assert sink.emitted == []


async def test_notify_hitl_pending_post_then_record_crash_returns_result() -> None:
    """Post succeeds, audit emit raises: the post stands (result returned), the
    emit failure is swallowed (the accepted crash-window residual), no row recorded."""
    notifier = _FakeNotifier(ts="9.9")
    sink = _FakeSink(raise_on_emit=RuntimeError("db down"))
    result = await _orch(notifier, sink).notify_hitl_pending(
        review_id=uuid4(),
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="t",
        findings=[],
    )
    assert result is not None
    assert result.ts == "9.9"
    assert sink.emitted == []


async def test_notify_hitl_pending_swallows_dedup_query_failure() -> None:
    """An infra failure in the pre-post dedup query fails closed (no post) and never
    raises into the caller."""
    notifier = _FakeNotifier()
    sink = _FakeSink(raise_on_query=RuntimeError("db down"))
    result = await _orch(notifier, sink).notify_hitl_pending(
        review_id=uuid4(),
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="t",
        findings=[],
    )
    assert result is None
    assert notifier.posts == []


async def test_notify_hitl_pending_propagates_is_eval() -> None:
    """is_eval threads onto the emitted event (eval isolation)."""
    notifier, sink = _FakeNotifier(), _FakeSink()
    await _orch(notifier, sink).notify_hitl_pending(
        review_id=uuid4(),
        is_eval=True,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="t",
        findings=[],
    )
    assert len(sink.emitted) == 1
    assert sink.emitted[0].is_eval is True


async def test_notify_review_posted_posts_and_records() -> None:
    """Clean non-gated FYI: one review_posted event + the routing counts in the text."""
    notifier, sink = _FakeNotifier(ts="5.5"), _FakeSink()
    rid = uuid4()
    result = await _orch(notifier, sink).notify_review_posted(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=12,
        posted_count=3,
        dashboard_only_count=1,
    )
    assert result is not None
    assert result.ts == "5.5"
    assert len(notifier.posts) == 1
    assert "3 posted" in notifier.posts[0]["text"]
    assert f"{_BASE_URL}/reviews/{rid}" in notifier.posts[0]["text"]

    assert len(sink.emitted) == 1
    assert sink.emitted[0].kind == "review_posted"
    assert sink.emitted[0].message_ts == "5.5"


async def test_notify_review_posted_skips_when_review_gated() -> None:
    """Mutual exclusion: a hitl_pending row means the review gated, so the FYI is
    suppressed (its terminal is the status mirror, not a review-posted message)."""
    rid = uuid4()
    notifier = _FakeNotifier()
    sink = _FakeSink(
        existing={(rid, "C1", "hitl_pending"): _slack_event(rid, "C1", "hitl_pending")}
    )
    result = await _orch(notifier, sink).notify_review_posted(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=12,
        posted_count=3,
        dashboard_only_count=1,
    )
    assert result is None
    assert notifier.posts == []
    assert sink.emitted == []


async def test_notify_review_posted_dedup_skips_when_already_posted() -> None:
    """A pre-existing review_posted row suppresses the re-post on replay."""
    rid = uuid4()
    notifier = _FakeNotifier()
    sink = _FakeSink(
        existing={(rid, "C1", "review_posted"): _slack_event(rid, "C1", "review_posted")}
    )
    result = await _orch(notifier, sink).notify_review_posted(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=12,
        posted_count=3,
        dashboard_only_count=1,
    )
    assert result is None
    assert notifier.posts == []
    assert sink.emitted == []


async def test_notify_review_posted_swallows_post_failure() -> None:
    """A transport failure on the FYI degrades to None with no audit row, no raise."""
    notifier = _FakeNotifier(raise_on_post=SlackChannelError("not_in_channel"))
    sink = _FakeSink()
    result = await _orch(notifier, sink).notify_review_posted(
        review_id=uuid4(),
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=12,
        posted_count=0,
        dashboard_only_count=0,
    )
    assert result is None
    assert len(notifier.posts) == 1  # the post was attempted; the transport raised
    assert sink.emitted == []


async def test_notify_hitl_pending_swallows_record_construction_failure() -> None:
    """A post returning a schema-invalid ts (empty) must NOT raise: the
    SlackNotificationEvent construction failure inside _record is swallowed like an
    emit failure (post stands, no row). Pins _record's self-contained no-raise."""
    notifier = _FakeNotifier(ts="")
    sink = _FakeSink()
    result = await _orch(notifier, sink).notify_hitl_pending(
        review_id=uuid4(),
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="t",
        findings=[],
    )
    assert len(notifier.posts) == 1
    assert result is not None  # the post happened; only the audit record failed
    assert sink.emitted == []


async def test_notify_hitl_pending_distinct_review_not_suppressed() -> None:
    """Dedup is keyed on review_id: a hitl_pending row for ONE review must not
    suppress a DIFFERENT review posting to the same channel."""
    rid_a, rid_b = uuid4(), uuid4()
    notifier = _FakeNotifier()
    sink = _FakeSink(
        existing={(rid_a, "C1", "hitl_pending"): _slack_event(rid_a, "C1", "hitl_pending")}
    )
    result = await _orch(notifier, sink).notify_hitl_pending(
        review_id=rid_b,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=8,
        pr_title="t",
        findings=[],
    )
    assert result is not None
    assert len(notifier.posts) == 1
    assert len(sink.emitted) == 1
    assert sink.emitted[0].review_id == rid_b


async def test_notify_review_posted_idempotent_across_two_calls() -> None:
    """review_posted replay idempotency end-to-end: the second identical call dedups
    against the first call's emitted row (emit-then-requery, not just seeding)."""
    notifier, sink = _FakeNotifier(), _FakeSink()
    orch, rid = _orch(notifier, sink), uuid4()
    first = await orch.notify_review_posted(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=12,
        posted_count=2,
        dashboard_only_count=0,
    )
    second = await orch.notify_review_posted(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=12,
        posted_count=2,
        dashboard_only_count=0,
    )
    assert first is not None
    assert second is None
    assert len(notifier.posts) == 1
    assert len(sink.emitted) == 1


async def test_notify_hitl_pending_re_posts_after_record_crash() -> None:
    """The accepted crash-window residual, end-to-end: a first call whose audit
    record crashes leaves no row, so a second call (replay) finds no dedup row and
    re-posts once — two posts, one recorded."""
    notifier = _FakeNotifier()
    sink = _FakeSink(fail_first_emit=True)
    orch, rid = _orch(notifier, sink), uuid4()
    findings = [_finding(FindingSeverity.HIGH)]
    first = await orch.notify_hitl_pending(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="t",
        findings=findings,
    )
    second = await orch.notify_hitl_pending(
        review_id=rid,
        is_eval=False,
        channel_id="C1",
        repo="acme/api",
        pr_number=7,
        pr_title="t",
        findings=findings,
    )
    assert first is not None
    assert second is not None
    assert len(notifier.posts) == 2  # the crash-window re-post
    assert len(sink.emitted) == 1  # only the second attempt recorded a row


# ---------------------------------------------------------------------------
# Status mirror (chat.update on the HITL card)
# ---------------------------------------------------------------------------


def _mirror_kwargs(review_id: UUID, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "review_id": review_id,
        "channel_id": "C1",
        "repo": "acme/api",
        "pr_number": 7,
        "pr_title": "Add webhook",
        "findings": [_finding(FindingSeverity.CRITICAL)],
        "reviewer_id": "alice",
        "posted_count": 3,
        "dashboard_only_count": 1,
    }
    base.update(overrides)
    return base


async def test_mirror_status_updates_the_recorded_hitl_card() -> None:
    """The mirror targets the `message_ts` + channel recorded on the original
    `hitl_pending` event — it edits the card that exists, never posts a new one."""
    review_id = uuid4()
    seeded = _slack_event(review_id, "C1", "hitl_pending")
    notifier, sink = (
        _FakeNotifier(),
        _FakeSink(existing={(review_id, "C1", "hitl_pending"): seeded}),
    )

    applied = await _orch(notifier, sink).mirror_status(**_mirror_kwargs(review_id))

    assert applied is True
    assert notifier.posts == []  # a mirror never posts
    assert len(notifier.updates) == 1
    assert notifier.updates[0]["channel"] == "C1"
    assert notifier.updates[0]["ts"] == seeded.message_ts


async def test_mirror_status_emits_no_audit_event() -> None:
    """The mirror is a reflection, not a new audited fact: no SlackNotificationEvent
    is emitted and the original row is never mutated (audit-events-append-only)."""
    review_id = uuid4()
    seeded = _slack_event(review_id, "C1", "hitl_pending")
    notifier, sink = (
        _FakeNotifier(),
        _FakeSink(existing={(review_id, "C1", "hitl_pending"): seeded}),
    )

    await _orch(notifier, sink).mirror_status(**_mirror_kwargs(review_id))

    assert sink.emitted == []


async def test_mirror_status_no_ops_when_review_never_gated() -> None:
    """No `hitl_pending` row → nothing to mirror. This is the non-gated review, and
    it is the guard that makes `mirror_status` the exact complement of
    `notify_review_posted` (which skips when the row DOES exist)."""
    notifier, sink = _FakeNotifier(), _FakeSink()

    applied = await _orch(notifier, sink).mirror_status(**_mirror_kwargs(uuid4()))

    assert applied is False
    assert notifier.updates == []


async def test_mirror_status_no_ops_when_channel_rotated() -> None:
    """The card was posted to C1; the install now resolves to C2. The lookup is
    channel-scoped, so the mirror finds no row and no-ops — it must never edit an
    unrelated message in the new channel."""
    review_id = uuid4()
    seeded = _slack_event(review_id, "C1", "hitl_pending")
    notifier, sink = (
        _FakeNotifier(),
        _FakeSink(existing={(review_id, "C1", "hitl_pending"): seeded}),
    )

    applied = await _orch(notifier, sink).mirror_status(
        **_mirror_kwargs(review_id, channel_id="C2")
    )

    assert applied is False
    assert notifier.updates == []


async def test_mirror_status_swallows_transport_failure() -> None:
    """Fire-and-forget: a `chat.update` failure leaves the card at its prior state
    and never raises into the graph."""
    review_id = uuid4()
    seeded = _slack_event(review_id, "C1", "hitl_pending")
    notifier = _FakeNotifier()
    sink = _FakeSink(existing={(review_id, "C1", "hitl_pending"): seeded})

    async def _boom(**_kwargs: Any) -> None:
        raise SlackChannelError("channel_not_found")

    notifier.update_message = _boom  # type: ignore[method-assign]

    assert await _orch(notifier, sink).mirror_status(**_mirror_kwargs(review_id)) is False


async def test_mirror_status_swallows_query_failure() -> None:
    """A dedup/lookup crash degrades to no mirror rather than propagating."""
    notifier = _FakeNotifier()
    sink = _FakeSink(raise_on_query=RuntimeError("db down"))

    assert await _orch(notifier, sink).mirror_status(**_mirror_kwargs(uuid4())) is False
    assert notifier.updates == []
