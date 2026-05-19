"""Tests for `outrider.agent.nodes.intake` — the webhook→graph node.

Covers the four behavior shapes specified by the intake-and-webhook spec:
  - Happy path: phase 1 list + phase 2 fetch + ChangedFile assembly +
    Command(goto='triage') return.
  - Size-gate skip: > 1000 lines OR > 30 files → status='skipped',
    Command(goto=END), no per-file events.
  - Per-file oversize: fetch returns None → FileExaminationEvent(skipped,
    OVERSIZED), file dropped from changed_files, intake still succeeds.
  - Failure mode: any exception → phase-end emitted FIRST, then
    status='failed' written, then re-raise.

Uses hand-rolled stubs for `github_factory`, `db_factory`, and the two
sinks; no Postgres or HTTP contact.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from langgraph.graph import END
from langgraph.types import Command

from outrider.agent.nodes.intake import intake
from outrider.agent.state import ReviewState
from outrider.audit.events import (  # noqa: TC001 — used in runtime list[X] annotations + isinstance assertions
    FileExaminationEvent,
    ReviewPhaseEvent,
)
from outrider.schemas.pr_context import PRContext

# ---------------------------------------------------------------------------
# Recording sinks (test fixtures only)
# ---------------------------------------------------------------------------


class _RecordingPhaseEventSink:
    """Captures every emit_phase call into a list for assertion."""

    def __init__(self) -> None:
        self.events: list[ReviewPhaseEvent] = []

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        self.events.append(event)


class _RecordingFileExaminationSink:
    """Captures every emit_file_examination call."""

    def __init__(self) -> None:
        self.events: list[FileExaminationEvent] = []

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Stub GitHub client + factory
# ---------------------------------------------------------------------------


@dataclass
class _StubFileMeta:
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None = None
    previous_filename: str | None = None


@dataclass
class _StubContentFile:
    """Like githubkit's ContentFile — has encoding + content."""

    encoding: str = "base64"
    content: str = ""


@dataclass
class _StubResponse:
    parsed_data: Any


class _StubReposAPI:
    """Returns a mapping of (path, ref) → bytes; missing entries → empty."""

    def __init__(self, content_by_key: dict[tuple[str, str], bytes]) -> None:
        self._content_by_key = content_by_key
        self.calls: list[tuple[str, str, str, str]] = []

    async def async_get_content(
        self, owner: str, repo: str, path: str, *, ref: str
    ) -> _StubResponse:
        self.calls.append((owner, repo, path, ref))
        content_bytes = self._content_by_key.get((path, ref))
        if content_bytes is None:
            # Simulate the no-content shape — `encoding="none"` returns None
            # downstream of fetch_file_content_at.
            return _StubResponse(parsed_data=_StubContentFile(encoding="none", content=""))
        return _StubResponse(
            parsed_data=_StubContentFile(
                encoding="base64",
                content=base64.b64encode(content_bytes).decode("ascii"),
            )
        )


class _StubPullsAPI:
    def __init__(self, files_metadata: list[_StubFileMeta]) -> None:
        self._files = files_metadata
        self.calls: list[tuple[str, str, int, dict[str, Any]]] = []

    async def async_list_files(
        self, owner: str, repo: str, pull_number: int, **kwargs: Any
    ) -> _StubResponse:
        self.calls.append((owner, repo, pull_number, kwargs))
        return _StubResponse(parsed_data=self._files)


class _StubRestAPI:
    def __init__(self, *, repos: _StubReposAPI, pulls: _StubPullsAPI) -> None:
        self.repos = repos
        self.pulls = pulls


class _StubGitHub:
    def __init__(
        self,
        *,
        files_metadata: list[_StubFileMeta],
        content_by_key: dict[tuple[str, str], bytes],
    ) -> None:
        self.rest = _StubRestAPI(
            repos=_StubReposAPI(content_by_key),
            pulls=_StubPullsAPI(files_metadata),
        )


def _stub_github_factory(gh: Any) -> Any:
    """Wrap any stub `GitHub`-like into a `Callable[[int], GitHub]` shape.

    Accepts `_StubGitHub`, `_FailingGitHub`, or any other test stub that
    duck-types as a githubkit `GitHub` for the intake call sites we
    exercise.
    """

    def factory(installation_id: int) -> Any:
        return gh

    return factory


# ---------------------------------------------------------------------------
# Stub async sessionmaker / session
# ---------------------------------------------------------------------------


@dataclass
class _RecordingSession:
    """Records every `execute` call without touching a real DB."""

    executed: list[Any] = field(default_factory=list)

    async def execute(self, stmt: Any) -> Any:
        self.executed.append(stmt)
        # Mimic SQLAlchemy's CursorResult enough for the call site
        return None

    async def commit(self) -> None:
        return None


class _StubSessionTransaction:
    """Mimics `session.begin()` async context manager."""

    def __init__(self, session: _RecordingSession) -> None:
        self.session = session

    async def __aenter__(self) -> _RecordingSession:
        return self.session

    async def __aexit__(self, *args: Any) -> None:
        return None


class _StubSession:
    def __init__(self, recording: _RecordingSession) -> None:
        self._recording = recording

    async def __aenter__(self) -> _RecordingSession:
        return self._recording

    async def __aexit__(self, *args: Any) -> None:
        return None

    def begin(self) -> _StubSessionTransaction:
        return _StubSessionTransaction(self._recording)


class _StubSessionFactory:
    """Returns the same _RecordingSession per call (for cross-call inspection)."""

    def __init__(self) -> None:
        self.recording = _RecordingSession()
        self.call_count = 0

    def __call__(self) -> Any:
        self.call_count += 1
        return _StubSession(self.recording)


# Per intake's `async with db_factory() as session, session.begin():`
# pattern, the session object's `__aenter__` returns a session that supports
# `session.begin()` as an async-context-manager AND `session.execute(...)`.
# To support that, we let the recording session itself answer both:


class _DualPurposeSession:
    """An object that is both the session itself AND the `begin()` source.

    Intake uses: `async with db_factory() as session, session.begin(): ...`
    The `db_factory()` is `_StubSessionFactory.__call__()` which returns
    something whose `__aenter__` returns `session`. Then `session.begin()`
    returns another async context manager.
    """

    def __init__(self, recording: _RecordingSession) -> None:
        self._recording = recording

    async def execute(self, stmt: Any) -> Any:
        return await self._recording.execute(stmt)

    def begin(self) -> _StubSessionTransaction:
        return _StubSessionTransaction(self._recording)

    async def __aenter__(self) -> _DualPurposeSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _StubSessionFactoryV2:
    def __init__(self) -> None:
        self.recording = _RecordingSession()
        self.call_count = 0

    def __call__(self) -> _DualPurposeSession:
        self.call_count += 1
        return _DualPurposeSession(self.recording)


# ---------------------------------------------------------------------------
# Seed state builder
# ---------------------------------------------------------------------------


def _build_state(*, total_additions: int = 5, total_deletions: int = 2) -> ReviewState:
    return ReviewState(
        review_id=uuid4(),
        received_at=datetime.now(UTC),
        pr_context=PRContext(
            installation_id=12345,
            owner="acme",
            repo="widgets",
            pr_number=42,
            base_sha="b" * 40,
            head_sha="h" * 40,
            pr_title="Test PR",
            pr_body=None,
            author="alice",
            total_additions=total_additions,
            total_deletions=total_deletions,
            changed_files=(),
        ),
    )


# ===========================================================================
# Happy path
# ===========================================================================


@pytest.mark.asyncio
async def test_happy_path_returns_command_to_triage() -> None:
    """One modified file: phase 1 returns metadata, phase 2 fetches
    base+head content, ChangedFile is constructed, Command(goto='triage')
    returned with new_pr_context."""
    state = _build_state()
    files = [
        _StubFileMeta(
            filename="src/example.py",
            status="modified",
            additions=5,
            deletions=2,
            patch="@@ -1 +1 @@\n-old\n+new\n",
        ),
    ]
    content = {
        ("src/example.py", "b" * 40): b"old\n",
        ("src/example.py", "h" * 40): b"new\n",
    }
    gh = _StubGitHub(files_metadata=files, content_by_key=content)
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    result = await intake(
        state,
        github_factory=_stub_github_factory(gh),
        db_factory=session_factory,  # type: ignore[arg-type]
        phase_event_sink=phase_sink,
        file_examination_sink=file_sink,
    )

    assert isinstance(result, Command)
    assert result.goto == "triage"
    # The update payload carries the enriched pr_context.
    update_payload = result.update
    assert update_payload is not None
    assert "pr_context" in update_payload
    new_ctx: PRContext = update_payload["pr_context"]
    assert len(new_ctx.changed_files) == 1
    cf = new_ctx.changed_files[0]
    assert cf.path == "src/example.py"
    assert cf.status == "modified"
    assert cf.content_base == "old\n"
    assert cf.content_head == "new\n"

    # Two phase events emitted (start + end).
    assert len(phase_sink.events) == 2
    assert [e.marker for e in phase_sink.events] == ["start", "end"]
    assert all(e.node_id == "intake" for e in phase_sink.events)

    # One FileExaminationEvent emitted (parse_status='clean').
    assert len(file_sink.events) == 1
    assert file_sink.events[0].parse_status == "clean"
    assert file_sink.events[0].file_path == "src/example.py"

    # No DB writes on the happy path.
    assert session_factory.call_count == 0


# ===========================================================================
# Size-gate skip
# ===========================================================================


@pytest.mark.asyncio
async def test_size_gate_lines_skips_to_end_without_per_file_events() -> None:
    """Total lines > 1000 → status='skipped' written, Command(goto=END),
    no per-file FileExaminationEvent (whole fan-out bypassed)."""
    state = _build_state(total_additions=600, total_deletions=500)  # 1100 total
    files = [
        _StubFileMeta(filename=f"f{i}.py", status="modified", additions=2, deletions=2)
        for i in range(3)
    ]
    gh = _StubGitHub(files_metadata=files, content_by_key={})
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    result = await intake(
        state,
        github_factory=_stub_github_factory(gh),
        db_factory=session_factory,  # type: ignore[arg-type]
        phase_event_sink=phase_sink,
        file_examination_sink=file_sink,
    )

    assert isinstance(result, Command)
    assert result.goto == END
    # update payload is None / empty — no pr_context enrichment on skip.
    assert result.update is None

    # Phase events still emitted (start + end).
    assert len(phase_sink.events) == 2

    # No per-file events — the fan-out was bypassed.
    assert file_sink.events == []

    # DB write happened (status='skipped').
    assert session_factory.call_count == 1


@pytest.mark.asyncio
async def test_size_gate_file_count_skips_to_end() -> None:
    """> 30 files → skip, same as the lines gate."""
    state = _build_state()  # only 7 total lines
    files = [
        _StubFileMeta(filename=f"f{i}.py", status="modified", additions=0, deletions=0)
        for i in range(31)
    ]
    gh = _StubGitHub(files_metadata=files, content_by_key={})
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    result = await intake(
        state,
        github_factory=_stub_github_factory(gh),
        db_factory=session_factory,  # type: ignore[arg-type]
        phase_event_sink=phase_sink,
        file_examination_sink=file_sink,
    )

    assert result.goto == END
    assert file_sink.events == []
    assert session_factory.call_count == 1


# ===========================================================================
# Per-file oversize → SkipReason.OVERSIZED
# ===========================================================================


@pytest.mark.asyncio
async def test_per_file_oversize_emits_skipped_event_drops_from_changed_files() -> None:
    """Per-file content cap exceeded: fetch returns None → emit
    FileExaminationEvent(parse_status='skipped', skip_reason=OVERSIZED),
    file dropped from changed_files. Intake still returns Command(goto='triage')
    because other files succeeded."""
    state = _build_state()
    files = [
        _StubFileMeta(
            filename="small.py",
            status="added",
            additions=5,
            deletions=0,
        ),
        _StubFileMeta(
            filename="huge.py",
            status="added",
            additions=999,
            deletions=0,
        ),
    ]
    content = {
        ("small.py", "h" * 40): b"def small():\n    pass\n",
        # huge.py NOT in content_by_key → stub returns encoding="none" →
        # fetch_file_content_at returns None → intake emits OVERSIZED.
    }
    gh = _StubGitHub(files_metadata=files, content_by_key=content)
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    result = await intake(
        state,
        github_factory=_stub_github_factory(gh),
        db_factory=session_factory,  # type: ignore[arg-type]
        phase_event_sink=phase_sink,
        file_examination_sink=file_sink,
    )

    assert result.goto == "triage"
    update_payload = result.update
    assert update_payload is not None
    new_ctx: PRContext = update_payload["pr_context"]
    assert len(new_ctx.changed_files) == 1
    assert new_ctx.changed_files[0].path == "small.py"

    # Two FileExaminationEvents: one clean (small.py), one skipped (huge.py).
    statuses = sorted((e.parse_status, e.file_path) for e in file_sink.events)
    assert statuses == [("clean", "small.py"), ("skipped", "huge.py")]


# ===========================================================================
# Per-file binary / malformed-UTF-8 content → SkipReason.OVERSIZED
# (FUP-033 will split out SkipReason.BINARY pending DECISIONS#018 amendment)
# ===========================================================================


@pytest.mark.asyncio
async def test_per_file_binary_content_skipped_not_silently_corrupted() -> None:
    """Binary content (NUL byte present) is classified as a skip — NOT
    silently corrupted into U+FFFD-replacement text and flowed to
    triage/analyze as 'clean'.

    `_classify_or_reserve_decode` checks `b'\\x00' in content_bytes`
    BEFORE the UTF-8 decode. Skip is emitted with `SkipReason.OVERSIZED`
    because DECISIONS#018 fixes the V1 enum to five values (no
    `BINARY`); the canonical amendment is tracked at FUP-033. The
    load-bearing assertion is the behavior (binary blobs skipped, not
    corrupted); the OVERSIZED routing is a canonical-conformant
    intermediate.
    """
    from outrider.ast_facts.models import SkipReason

    state = _build_state()
    files = [
        _StubFileMeta(filename="text.py", status="added", additions=2, deletions=0),
        _StubFileMeta(filename="blob.bin", status="added", additions=1, deletions=0),
    ]
    content = {
        ("text.py", "h" * 40): b"def small():\n    pass\n",
        # PDF-shaped binary: %PDF header + NUL byte + binary payload.
        # NUL is the definitive marker the skip-classifier checks.
        ("blob.bin", "h" * 40): b"%PDF-1.4\n\x00\x00\xff\xfe\x80\x81binary-junk",
    }
    gh = _StubGitHub(files_metadata=files, content_by_key=content)
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    result = await intake(
        state,
        github_factory=_stub_github_factory(gh),
        db_factory=session_factory,  # type: ignore[arg-type]
        phase_event_sink=phase_sink,
        file_examination_sink=file_sink,
    )

    assert result.goto == "triage"
    update_payload = result.update
    assert update_payload is not None
    new_ctx: PRContext = update_payload["pr_context"]
    # Only the text file survives; the binary blob is dropped.
    assert [cf.path for cf in new_ctx.changed_files] == ["text.py"]

    # Two FileExaminationEvents: clean (text.py), skipped+OVERSIZED (blob.bin).
    by_path = {e.file_path: e for e in file_sink.events}
    assert by_path["text.py"].parse_status == "clean"
    assert by_path["blob.bin"].parse_status == "skipped"
    assert by_path["blob.bin"].skip_reason == SkipReason.OVERSIZED


@pytest.mark.asyncio
async def test_per_file_malformed_utf8_emits_skip_not_corruption() -> None:
    """File with no NUL byte but invalid UTF-8 sequence is also routed
    to skip (not flowed as 'clean'). Both binary blobs and partially-
    corrupted text get the same operator-visible outcome — neither
    reaches the LLM as clean content.

    Skip reason is OVERSIZED for the same DECISIONS#018-conformance
    reason as the binary case (see FUP-033 for the canonical amendment).
    """
    from outrider.ast_facts.models import SkipReason

    state = _build_state()
    files = [
        _StubFileMeta(filename="garbled.txt", status="added", additions=1, deletions=0),
    ]
    content = {
        # No NUL byte, but the second byte (0x80) is an invalid UTF-8
        # continuation byte without a leading 0xC0/0xE0/0xF0 — strict
        # UTF-8 decode raises UnicodeDecodeError → classifier returns
        # SkipReason.OVERSIZED (binary/malformed routed through the
        # same reason pending FUP-033's canonical-amendment).
        ("garbled.txt", "h" * 40): b"a\x80\x81b",
    }
    gh = _StubGitHub(files_metadata=files, content_by_key=content)
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    result = await intake(
        state,
        github_factory=_stub_github_factory(gh),
        db_factory=session_factory,  # type: ignore[arg-type]
        phase_event_sink=phase_sink,
        file_examination_sink=file_sink,
    )

    assert result.goto == "triage"
    assert result.update is not None
    assert result.update["pr_context"].changed_files == ()
    assert len(file_sink.events) == 1
    skipped = file_sink.events[0]
    assert skipped.parse_status == "skipped"
    assert skipped.skip_reason == SkipReason.OVERSIZED


# ===========================================================================
# `_ByteBudget` atomicity (audit-the-audit round MEDIUM)
# ===========================================================================


@pytest.mark.asyncio
async def test_byte_budget_try_reserve_is_atomic_under_concurrent_calls() -> None:
    """`_ByteBudget.try_reserve` MUST be atomic: under N concurrent
    callers each requesting `cap - 1` bytes, exactly one succeeds and
    the rest are denied. Tests the asyncio.Lock-then-check-then-set
    sequence directly — the higher-level intake tests admit non-
    determinism under the semaphore + stub GitHub timing, so the lock
    invariant could regress without the intake tests catching it.

    A regression that removed `async with self._lock:` and made
    `try_reserve` a bare check-and-increment would pass the existing
    `test_aggregate_bytes_cap_emits_oversized_when_total_exceeded`
    because that test serializes naturally under the zero-latency
    stub. This test explicitly fans out N parallel reservations to
    surface the race.
    """
    from outrider.agent.nodes.intake import _ByteBudget

    cap = 1000
    budget = _ByteBudget(cap=cap)

    n_callers = 16
    request_size = cap - 1  # each call alone fits; any two overflow

    results = await asyncio.gather(*[budget.try_reserve(request_size) for _ in range(n_callers)])

    # Exactly one caller admitted; the rest denied.
    assert sum(results) == 1, (
        f"Expected exactly 1 reservation to succeed, got {sum(results)}. "
        f"Atomicity broken: multiple callers passed the check-and-set."
    )

    # The accumulator never overshoots the cap.
    # (Accessing the private `_used` is acceptable in a test that
    # specifically guards the lock's invariant.)
    assert budget._used <= cap


@pytest.mark.asyncio
async def test_byte_budget_release_returns_bytes_to_pool() -> None:
    """`_ByteBudget.release(n)` decrements `_used` by `n` so a
    previously-reserved-but-rolled-back chunk can be re-used by later
    files.

    Pins the rollback path used by the two-sided fetch branches
    (modified / renamed) when one side classifies clean and reserves
    budget but the other side fails — without release, the clean
    side's reservation would crowd out later valid files.
    """
    from outrider.agent.nodes.intake import _ByteBudget

    budget = _ByteBudget(cap=1000)

    # Reserve 600, release 600 — budget fully restored.
    assert await budget.try_reserve(600) is True
    assert budget._used == 600
    await budget.release(600)
    assert budget._used == 0

    # A subsequent 1000-byte reservation must fit (would NOT fit if
    # the prior 600 had stuck).
    assert await budget.try_reserve(1000) is True

    # `release` clamps at zero — releasing more than was reserved
    # leaves the accumulator at 0, not negative.
    await budget.release(99_999)
    assert budget._used == 0


@pytest.mark.asyncio
async def test_modified_file_releases_clean_side_when_other_side_binary() -> None:
    """Modified file with clean base + binary head: the clean base's
    byte reservation MUST be released back to the budget before the
    file is dropped, otherwise later valid files are skipped as
    OVERSIZED due to phantom accounting.

    Pins the rollback wiring in the modified branch of
    `_process_one_file`.
    """
    from outrider.ast_facts.models import SkipReason

    state = _build_state()
    files = [
        _StubFileMeta(filename="dirty.py", status="modified", additions=1, deletions=1),
        _StubFileMeta(filename="clean.py", status="added", additions=1, deletions=0),
    ]
    # dirty.py: clean base, binary head (NUL byte in head)
    # clean.py: small clean text
    payload_base_clean = b"def f(): pass\n"  # clean text, valid UTF-8
    payload_head_binary = b"binary\x00content"  # NUL byte → BINARY skip
    payload_clean = b"def g(): pass\n"

    content = {
        ("dirty.py", "b" * 40): payload_base_clean,  # base_sha
        ("dirty.py", "h" * 40): payload_head_binary,  # head_sha
        ("clean.py", "h" * 40): payload_clean,
    }
    gh = _StubGitHub(files_metadata=files, content_by_key=content)
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    # Drop the aggregate cap to ONE BYTE below the sum of both clean
    # files' sizes. Math:
    #   - WITH release: dirty.py reserves 14 (base) then releases 14
    #     when head fails NUL-check → used=0. clean.py reserves 14 →
    #     used=14 ≤ 27. Admitted.
    #   - WITHOUT release: dirty.py reserves 14 and keeps it → used=14.
    #     clean.py tries to reserve 14 → 14+14=28 > 27. REJECTED.
    # The `-1` makes the test fail loudly under a `release()`-omitted
    # regression. Without it (cap = 14 + 14 = 28), both files fit
    # regardless of whether release ran — the test would be vacuous.
    from outrider.agent.nodes import intake as intake_mod

    monkeypatch_cap = len(payload_base_clean) + len(payload_clean) - 1
    original_cap = intake_mod._TOTAL_DECODED_BYTES_CAP
    intake_mod._TOTAL_DECODED_BYTES_CAP = monkeypatch_cap
    try:
        result = await intake(
            state,
            github_factory=_stub_github_factory(gh),
            db_factory=session_factory,  # type: ignore[arg-type]
            phase_event_sink=phase_sink,
            file_examination_sink=file_sink,
        )
    finally:
        intake_mod._TOTAL_DECODED_BYTES_CAP = original_cap

    assert result.goto == "triage"
    assert result.update is not None
    new_ctx: PRContext = result.update["pr_context"]

    # `clean.py` must be admitted as clean (it would NOT be if
    # `dirty.py`'s base reservation hadn't been released).
    paths = [cf.path for cf in new_ctx.changed_files]
    assert paths == ["clean.py"]

    # Audit events: `dirty.py` skipped+OVERSIZED (binary), `clean.py` clean.
    by_path = {e.file_path: e for e in file_sink.events}
    assert by_path["dirty.py"].parse_status == "skipped"
    assert by_path["dirty.py"].skip_reason == SkipReason.OVERSIZED
    assert by_path["clean.py"].parse_status == "clean"


# ===========================================================================
# Aggregate decoded-bytes cap
# ===========================================================================


@pytest.mark.asyncio
async def test_aggregate_bytes_cap_emits_oversized_when_total_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the aggregate decoded-bytes accumulator would overshoot the
    intake-wide cap, the offending file is skipped with OVERSIZED — NOT
    the whole intake. Files admitted under the cap still flow as clean.

    Drops the cap to a small value via monkeypatch so two ~600-byte
    files trigger the overshoot deterministically.
    """
    from outrider.agent.nodes import intake as intake_mod
    from outrider.ast_facts.models import SkipReason

    # Tiny cap so we can trigger overshoot with small test fixtures.
    monkeypatch.setattr(intake_mod, "_TOTAL_DECODED_BYTES_CAP", 1000)

    state = _build_state()
    files = [
        _StubFileMeta(filename="a.py", status="added", additions=1, deletions=0),
        _StubFileMeta(filename="b.py", status="added", additions=1, deletions=0),
    ]
    # Each file is 600 bytes — first fits (used=600 ≤ 1000), second
    # would push to 1200 > 1000 and gets dropped.
    payload_a = b"a" * 600
    payload_b = b"b" * 600
    content = {
        ("a.py", "h" * 40): payload_a,
        ("b.py", "h" * 40): payload_b,
    }
    gh = _StubGitHub(files_metadata=files, content_by_key=content)
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    result = await intake(
        state,
        github_factory=_stub_github_factory(gh),
        db_factory=session_factory,  # type: ignore[arg-type]
        phase_event_sink=phase_sink,
        file_examination_sink=file_sink,
    )

    assert result.goto == "triage"
    assert result.update is not None
    new_ctx: PRContext = result.update["pr_context"]
    # Exactly one file admitted (whichever fetched first under semaphore);
    # other is dropped with OVERSIZED. asyncio.gather doesn't guarantee
    # order, so assert on counts + reasons, not identities.
    assert len(new_ctx.changed_files) == 1
    statuses = [(e.parse_status, e.skip_reason) for e in file_sink.events]
    clean_count = sum(1 for ps, _ in statuses if ps == "clean")
    oversized_count = sum(
        1 for ps, sr in statuses if ps == "skipped" and sr == SkipReason.OVERSIZED
    )
    assert clean_count == 1
    assert oversized_count == 1


@pytest.mark.asyncio
async def test_unknown_status_logged_and_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file with an unknown GitHub status (e.g., `"copied"`, or a
    future GitHub-added status) → log warning + drop file + NO
    FileExaminationEvent emitted. Forward-compat behavior the code
    documents at intake.py:644-649. Without this test the branch is
    uncovered; a regression that crashed the whole intake on unknown
    status would only fail when GitHub adds a new status in production.
    """
    import logging

    state = _build_state()
    files = [
        _StubFileMeta(filename="ok.py", status="added", additions=1, deletions=0),
        _StubFileMeta(filename="copied.py", status="copied", additions=1, deletions=0),
    ]
    content = {("ok.py", "h" * 40): b"def f(): pass\n"}
    gh = _StubGitHub(files_metadata=files, content_by_key=content)
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    with caplog.at_level(logging.WARNING, logger="outrider.agent.nodes.intake"):
        result = await intake(
            state,
            github_factory=_stub_github_factory(gh),
            db_factory=session_factory,  # type: ignore[arg-type]
            phase_event_sink=phase_sink,
            file_examination_sink=file_sink,
        )

    assert result.goto == "triage"
    assert result.update is not None
    new_ctx: PRContext = result.update["pr_context"]

    # `copied.py` is dropped — only `ok.py` survives.
    paths = [cf.path for cf in new_ctx.changed_files]
    assert paths == ["ok.py"]

    # NO FileExaminationEvent emitted for `copied.py` (the unknown-status
    # branch deliberately does NOT emit; per current behavior the audit
    # invisibility is documented at intake.py:644-649).
    file_paths_with_events = {e.file_path for e in file_sink.events}
    assert "copied.py" not in file_paths_with_events
    assert "ok.py" in file_paths_with_events

    # Warning log fires.
    skip_logs = [r for r in caplog.records if "unknown file status" in r.getMessage()]
    assert len(skip_logs) == 1
    assert "copied" in skip_logs[0].getMessage()


@pytest.mark.asyncio
async def test_binary_blob_does_not_consume_budget_before_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classify-then-reserve order is the load-bearing invariant of
    `_classify_or_reserve_decode` (intake.py docstring lines 757-760):
    binary/malformed bytes must NOT consume the aggregate text budget,
    or a single binary blob in an early file would starve later valid
    files into spurious OVERSIZED skips.

    Existing tests pin the OUTCOME of binary skip; this one pins the
    ORDER. Setup: cap=60, binary=55 NULs, text=6 bytes valid UTF-8.
    The regression-revealing case is when binary_size + text_size > cap
    but text_size alone fits.

      - WITH classify-first (current): binary fails NUL check without
        reserving → used=0. Text reserves 6 → used=6 ≤ 60. Admitted.
      - WITH reserve-first (regression): binary reserves 55 → used=55.
        Text tries to reserve 6 → 55+6=61 > 60. REJECTED.
    """
    from outrider.agent.nodes import intake as intake_mod
    from outrider.ast_facts.models import SkipReason

    monkeypatch.setattr(intake_mod, "_TOTAL_DECODED_BYTES_CAP", 60)

    state = _build_state()
    files = [
        _StubFileMeta(filename="binary.py", status="added", additions=1, deletions=0),
        _StubFileMeta(filename="text.py", status="added", additions=1, deletions=0),
    ]
    payload_binary = b"\x00" * 55  # 55 NULs — fails the NUL byte check
    payload_text = b"x = 1\n" * 1  # 6 bytes valid UTF-8

    content = {
        ("binary.py", "h" * 40): payload_binary,
        ("text.py", "h" * 40): payload_text,
    }
    gh = _StubGitHub(files_metadata=files, content_by_key=content)
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    result = await intake(
        state,
        github_factory=_stub_github_factory(gh),
        db_factory=session_factory,  # type: ignore[arg-type]
        phase_event_sink=phase_sink,
        file_examination_sink=file_sink,
    )

    assert result.goto == "triage"
    assert result.update is not None
    new_ctx: PRContext = result.update["pr_context"]

    # text.py MUST be admitted as clean. If classify-then-reserve
    # regressed to reserve-then-classify, the binary's 55 bytes would
    # have phantom-reserved budget and text.py would be skipped as
    # OVERSIZED at 55 + 6 = 61 > 60.
    paths = [cf.path for cf in new_ctx.changed_files]
    assert paths == ["text.py"], (
        "text.py was not admitted — binary.py's bytes consumed the "
        "aggregate budget. classify-then-reserve order regressed: a "
        "binary blob is now starving valid text files from the budget."
    )

    by_path = {e.file_path: e for e in file_sink.events}
    assert by_path["binary.py"].parse_status == "skipped"
    assert by_path["binary.py"].skip_reason == SkipReason.OVERSIZED
    assert by_path["text.py"].parse_status == "clean"


# ===========================================================================
# Failure mode: re-raise after status='failed'
# ===========================================================================


class _FailingPulls:
    """async_list_files always raises."""

    async def async_list_files(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated GitHub failure")


class _FailingGitHub:
    def __init__(self) -> None:
        self.rest = _StubRestAPI(
            repos=_StubReposAPI({}),
            pulls=_FailingPulls(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_intake_failure_reraises_after_status_write() -> None:
    """Exception during fetch → phase-end emitted FIRST, then
    status='failed' written, then exception re-raised."""
    state = _build_state()
    gh = _FailingGitHub()
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    with pytest.raises(RuntimeError, match="simulated GitHub failure"):
        await intake(
            state,
            github_factory=_stub_github_factory(gh),
            db_factory=session_factory,  # type: ignore[arg-type]
            phase_event_sink=phase_sink,
            file_examination_sink=file_sink,
        )

    # Phase events: start + end (both emitted before raise propagates).
    assert len(phase_sink.events) == 2
    assert [e.marker for e in phase_sink.events] == ["start", "end"]

    # Status='failed' write happened.
    assert session_factory.call_count == 1
    # No FileExaminationEvent — phase 2 was never reached.
    assert file_sink.events == []


@pytest.mark.asyncio
async def test_intake_status_write_failure_preserves_original_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If `_set_review_status` itself raises during the failure handler
    (e.g., DB connection lost), the ORIGINAL intake exception (here
    `RuntimeError("simulated GitHub failure")`) is re-raised — not the
    SQLAlchemy error from the cleanup.

    Without the exception-preserving try/except around
    `_set_review_status`, the operator would see the cleanup error and
    chase the wrong root cause.
    """
    import logging

    state = _build_state()
    gh = _FailingGitHub()
    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()

    class _StatusWriteFailingFactory:
        """db_factory that raises on every session open — simulates a
        DB connection loss mid-failure-handling."""

        def __init__(self) -> None:
            self.call_count = 0

        def __call__(self) -> Any:
            self.call_count += 1
            msg = "simulated db error during status write"
            raise OSError(msg)

    failing_factory = _StatusWriteFailingFactory()

    # The ORIGINAL exception (RuntimeError) re-raises — NOT the OSError
    # from `_set_review_status`. If FUP-032's fix regresses, the
    # assertion would catch the OSError as the chained `__cause__`
    # rather than the original RuntimeError.
    with (
        caplog.at_level(logging.ERROR, logger="outrider.agent.nodes.intake"),
        pytest.raises(RuntimeError, match="simulated GitHub failure"),
    ):
        await intake(
            state,
            github_factory=_stub_github_factory(gh),
            db_factory=failing_factory,  # type: ignore[arg-type]
            phase_event_sink=phase_sink,
            file_examination_sink=file_sink,
        )

    # The status-write attempt was made (and failed loud in logs).
    assert failing_factory.call_count == 1

    # The status-write failure was logged with the expected message
    # shape so an operator searching for stuck-running diagnoses can
    # find the cleanup gap.
    status_write_log_records = [
        r for r in caplog.records if "status='failed' write failed during failure" in r.getMessage()
    ]
    assert len(status_write_log_records) == 1


# ===========================================================================
# TaskGroup phase-2 failure (audit-the-audit round HIGH)
# ===========================================================================


@pytest.mark.asyncio
async def test_intake_phase2_failure_unwraps_taskgroup_exception() -> None:
    """A per-file fetch failure during phase-2 TaskGroup execution:
      - cancels sibling tasks (no FileExaminationEvent after phase-end);
      - unwraps the ExceptionGroup so the outer handler sees the original
        cause, NOT a wrapping group;
      - prefers the root-cause exception over a sibling CancelledError;
      - reaches the failure handler (phase-end emitted, status='failed').

    Pins the audit-the-audit round HIGH finding: pre-existing failure
    tests use `_FailingGitHub.async_list_files` which raises in phase-1
    BEFORE the TaskGroup is created. The TaskGroup body + ExceptionGroup
    unwrap was completely untested. A regression breaking the unwrap
    (e.g., `eg.exceptions[0]` → `eg.args`) wouldn't fail any other test.
    """
    state = _build_state()
    files = [
        _StubFileMeta(filename="ok.py", status="added", additions=1, deletions=0),
        _StubFileMeta(filename="boom.py", status="added", additions=1, deletions=0),
    ]
    # `ok.py` content is fine; `boom.py` fetch raises a typed error
    # mid-fan-out. The stub Repos API raises on the second file by
    # synthesizing an exception in `async_get_content`.

    class _BoomReposAPI:
        """Returns valid content for `ok.py`, raises a specific
        RuntimeError for `boom.py`. Mimics a real per-file fetch failure
        (e.g., a transient githubkit HTTPStatusError on one path)."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def async_get_content(
            self,
            owner: str,
            repo: str,
            path: str,
            *,
            ref: str,  # noqa: ARG002
        ) -> Any:
            self.calls.append(path)
            if path == "boom.py":
                msg = "simulated per-file fetch failure"
                raise RuntimeError(msg)
            # Return inline-base64 content for ok.py.
            return _StubResponse(
                parsed_data=_StubContentFile(
                    encoding="base64",
                    content=base64.b64encode(b"def ok(): pass\n").decode("ascii"),
                )
            )

    boom_gh = _StubGitHub(files_metadata=files, content_by_key={})
    boom_gh.rest = _StubRestAPI(  # type: ignore[assignment]
        repos=_BoomReposAPI(),  # type: ignore[arg-type]
        pulls=_StubPullsAPI(files),
    )

    phase_sink = _RecordingPhaseEventSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()

    # The ORIGINAL exception (RuntimeError "simulated per-file fetch
    # failure") propagates out — NOT an ExceptionGroup, NOT a
    # CancelledError, NOT a wrapping container. If the TaskGroup unwrap
    # regresses, pytest.raises(RuntimeError, match=...) fails because
    # the outer type would be ExceptionGroup or a CancelledError-shaped
    # sibling.
    with pytest.raises(RuntimeError, match="simulated per-file fetch failure"):
        await intake(
            state,
            github_factory=_stub_github_factory(boom_gh),
            db_factory=session_factory,  # type: ignore[arg-type]
            phase_event_sink=phase_sink,
            file_examination_sink=file_sink,
        )

    # Phase events: start + end (failure handler emitted end before
    # re-raising the unwrapped root cause).
    assert [e.marker for e in phase_sink.events] == ["start", "end"]

    # CRITICAL — no FileExaminationEvent emitted AFTER phase-end. The
    # TaskGroup cancellation discipline + ExceptionGroup unwrap is the
    # `phase-events-bound-work` invariant. Pre-TaskGroup gather allowed
    # sibling tasks to keep running and emit events after the failure-
    # path phase-end; TaskGroup forbids this. We can't directly observe
    # "after phase-end" ordering without timestamps, but we CAN assert
    # the count: at most one FileExaminationEvent from `ok.py`, NEVER
    # one from `boom.py` (which raised before reaching the emit call).
    file_paths = [e.file_path for e in file_sink.events]
    assert "boom.py" not in file_paths, (
        "boom.py raised in fetch — must NOT produce a FileExaminationEvent "
        "(the per-file `_process_one_file` body raises before emit)"
    )

    # status='failed' write happened.
    assert session_factory.call_count == 1


# ===========================================================================
# Phase-start emit failure (multi-lens audit at commit 20e4b62 HIGH)
# ===========================================================================


@pytest.mark.asyncio
async def test_intake_phase_start_emit_failure_does_not_emit_orphan_phase_end() -> None:
    """A persister failure on the FIRST emit (phase-start) must NOT cause
    the failure handler to emit a phase-end. Without the
    `phase_start_persisted` gate, the cleanup unconditionally emits
    phase-end and produces an end-only marker (durable phase-end with
    no matching phase-start) — equally destructive to
    `phase-events-bound-work` replay semantics as the orphan-start case.

    Behavior under the gate:
      - phase-start emit raises → outer try catches in failure handler
      - failure handler sees phase_start_persisted=False → skips phase-end
      - status='failed' write still happens
      - original phase-start exception propagates

    A regression that drops the gate would produce `events == [<end>]`
    instead of `events == []`.
    """
    state = _build_state()

    class _PhaseStartFailingSink:
        """emit_phase raises on the first call (phase-start), succeeds
        on any subsequent call. Tracks all attempts."""

        def __init__(self) -> None:
            self.attempts: list[ReviewPhaseEvent] = []
            self.persisted: list[ReviewPhaseEvent] = []

        async def emit_phase(self, event: ReviewPhaseEvent) -> None:
            self.attempts.append(event)
            if len(self.attempts) == 1:
                msg = "simulated persister failure on phase-start emit"
                raise RuntimeError(msg)
            self.persisted.append(event)

    phase_sink = _PhaseStartFailingSink()
    file_sink = _RecordingFileExaminationSink()
    session_factory = _StubSessionFactoryV2()
    # github_factory shape doesn't matter — intake fails before invoking it.
    gh = _FailingGitHub()

    with pytest.raises(RuntimeError, match="simulated persister failure on phase-start emit"):
        await intake(
            state,
            github_factory=_stub_github_factory(gh),
            db_factory=session_factory,  # type: ignore[arg-type]
            phase_event_sink=phase_sink,
            file_examination_sink=file_sink,
        )

    # One attempted emit (the failed phase-start); zero persisted.
    assert len(phase_sink.attempts) == 1
    assert phase_sink.attempts[0].marker == "start"
    assert phase_sink.persisted == [], (
        "phase-end was emitted despite phase-start emit failing — "
        "`phase_start_persisted` gate regressed; orphan end-only marker would land."
    )

    # status='failed' write still happened (cleanup is independent of phase-end).
    assert session_factory.call_count == 1
    # No file events — intake never reached phase-2.
    assert file_sink.events == []
