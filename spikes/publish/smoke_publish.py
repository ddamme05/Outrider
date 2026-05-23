# Direct-invoke smoke harness for the V1 publish node.
"""Exercise `outrider.agent.nodes.publish.publish(...)` end-to-end.

Two modes:

- **mock** (default; no external deps): stub publisher + recording sinks.
  Asserts that the eight-step pre-flight, routing, eligibility gate,
  attempt emission, and idempotency check all flow correctly under
  realistic ReviewState shapes. Runs in under a second; suitable for
  CI/local sanity.

- **live** (`--apply`; requires real GitHub App + Postgres): real
  `GitHubKitPublisher` + real `AuditPersister`. Posts a single review
  comment to a hard-allowlisted PR on `ddamme05/outrider-smoke-test`.
  Asserts pillar 1 (`PublishResult.success` shape) + pillar 2 (GitHub
  re-query via body-marker matcher returns the just-posted review).
  Pillar 3 (audit-row count + payload verification) is currently
  `[SKIP]` — `AuditPersister.emit_*` calls completing without raising
  is the V1 signal; row-count / payload-shape verification against
  `audit_events` is deferred to FUP-070. Then re-invokes publish() with
  the same `review_id` to assert the intra-Outrider idempotency path
  returns `PublishResult.idempotently_skipped` and NO duplicate comment
  posts.

This harness is the empirical validation of the publish path the unit
suite (1963 tests, stub publisher) cannot give: it proves githubkit
actually accepts our request shape, the body marker round-trips through
`GET /pulls/{n}/reviews`, the `AuditPersister.emit_*` calls do not
raise against real Postgres, and the FUP-064 intra-Outrider idempotency
check actually fires on a re-run. It does NOT prove that audit-event
rows landed with the expected per-event-type counts or payload content
(FUP-070).

Guard rails (per the multi-lens design audit):

- `--repo` is hardcoded-allowlisted; refuses any value other than
  `ddamme05/outrider-smoke-test`.
- `TEST_DATABASE_URL` must point at port 5433 with a name containing
  `outrider_test` — mirrors `tests/integration/conftest.py`'s guard.
- All synthetic findings are severity ∈ {MEDIUM, LOW, INFO} with
  `original_severity=None` — the V1 eligibility gate withholds
  everything else, which would silently yield "ran clean" with zero
  comments posted (the canonical false-success bug class).
- `is_eval=True` on every constructed `ReviewState` so harness writes
  don't pollute production dashboard queries (per `docs/testing.md`).
- `_BODY_MARKER_TEMPLATE` is imported from `agent/nodes/publish.py`,
  not re-derived — if the production template changes, the harness
  follows automatically.
- Per-run `cleanup_manifest.jsonl` records every `(timestamp,
  github_review_id)` the harness posts (live mode only); operator
  uses this for batch dismissal. Append-only, gitignored.

Out of scope for V1:

- Full-graph smoke (`await graph.ainvoke(seed_state)`) requires LLM
  credentials + a real PR with model-eligible findings; the four-node
  graph runs intake → triage → analyze → publish, and analyze needs
  a Sonnet call. Tracked separately.
- The `IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD` branch (Step 6 at
  `publish.py:284`) requires a crash-after-success scenario; deferred
  to a future harness extension.
- Automatic teardown of posted GitHub reviews: GitHub doesn't expose
  bulk-delete and submitted comment-reviews persist. Operator runs a
  separate cleanup pass against `cleanup_manifest.jsonl`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

# Sanitizer needs the truncation-marker HMAC secret to be set; set a
# harness-scoped default if the operator didn't (only fires when the
# env var is absent — operator-set values win).
os.environ.setdefault("OUTRIDER_TRUNCATION_HMAC_SECRET", "spikes-publish-smoke-secret")

from outrider.agent.nodes.publish import _BODY_MARKER_TEMPLATE, publish
from outrider.audit.events import (
    PublishAttemptEvent,
    PublishAttemptOutcome,
    PublishEligibility,
    PublishEligibilityEvent,
    PublishEvent,
    PublishRoutingEvent,
    ReviewPhaseEvent,
    compute_finding_content_hash,
)
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION, SEVERITY_POLICY
from outrider.schemas import (
    ChangedFile,
    GitHubReviewCreated,
    PRContext,
    PublishResult,
    ReviewFinding,
    ReviewState,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Hard-allowlisted publish targets.
#
# Adversarial-lens concern: a default repo string + env-var fallback is
# the canonical "operator typo posts to production" footgun. The
# allowlist refuses anything else, even with an explicit --repo flag.
# Adding a new target requires a code change (visible in PR review).
# ---------------------------------------------------------------------------
_REPO_ALLOWLIST: frozenset[tuple[str, str]] = frozenset({("ddamme05", "outrider-smoke-test")})


# ---------------------------------------------------------------------------
# Recording stubs (ported from tests/unit/test_publish_node_end_to_end.py)
# ---------------------------------------------------------------------------


class _RecordingPhaseEventSink:
    def __init__(self) -> None:
        self.events: list[ReviewPhaseEvent] = []

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        self.events.append(event)


class _RecordingPublishEventSink:
    """In-memory `PublishEventSink` for mock mode.

    Records every emit; serves `prior_publish_event` from a slot the
    harness sets between the two invocations so the second call hits
    the intra-Outrider idempotency path.
    """

    def __init__(self) -> None:
        self.routing: list[PublishRoutingEvent] = []
        self.eligibility: list[PublishEligibilityEvent] = []
        self.attempts: list[PublishAttemptEvent] = []
        self.results: list[PublishEvent] = []
        self.prior_publish_event: PublishEvent | None = None
        self.query_calls: list[UUID] = []

    async def emit_publish_routing(self, event: PublishRoutingEvent) -> None:
        self.routing.append(event)

    async def emit_publish_eligibility(self, event: PublishEligibilityEvent) -> None:
        self.eligibility.append(event)

    async def emit_publish_attempt(self, event: PublishAttemptEvent) -> None:
        self.attempts.append(event)

    async def emit_publish_result(self, event: PublishEvent) -> None:
        self.results.append(event)

    async def query_prior_publish_event(self, review_id: UUID) -> PublishEvent | None:
        self.query_calls.append(review_id)
        return self.prior_publish_event


class _StubPublisher:
    """Mock-mode publisher; returns canned `GitHubReviewCreated`.

    Always returns `None` from `find_existing_review_on_head_sha` — the
    external-record idempotency branch (publish.py Step 6) is out of
    scope for V1 (see module docstring). A future harness extension that
    exercises that branch should subclass and override, not flip a
    constructor arg.
    """

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.find_calls: list[dict[str, Any]] = []

    async def create_review(self, **kwargs: Any) -> GitHubReviewCreated:
        self.create_calls.append(kwargs)
        return GitHubReviewCreated(github_review_id=42, comments_posted=len(kwargs["comments"]))

    async def find_existing_review_on_head_sha(self, **kwargs: Any) -> int | None:
        self.find_calls.append(kwargs)
        return None


def _stub_github_factory(installation_id: int) -> Any:  # noqa: ARG001
    return object()


# ---------------------------------------------------------------------------
# Fixture builders. Severity is derived via `lookup_severity` so a drift
# between FindingType and SEVERITY_POLICY surfaces as a fixture-time
# failure rather than a green run against stale policy data.
# ---------------------------------------------------------------------------


def _make_changed_file(
    *,
    path: str,
    content_head: str | None,
    content_base: str | None = None,
    patch: str,
    status: str = "added",
    previous_path: str | None = None,
) -> ChangedFile:
    """Construct a `ChangedFile` for the smoke harness.

    `ChangedFile`'s validator enforces status-specific content-presence:
      - `added`:    content_base=None,    content_head=<str>
      - `modified`: content_base=<str>,   content_head=<str>
      - `removed`:  content_base=<str>,   content_head=None
      - `renamed`:  content_base=<str>,   content_head=<str>,
                    previous_path=<str>

    Caller is responsible for fetching the right content shape per status
    before invoking — the live-mode flow in `_run_live_mode` branches on
    `target.status` and fetches base content from `previous_filename`
    (renamed) or `path` (modified/removed) at `base_sha`.
    """
    additions = sum(
        1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    deletions = sum(
        1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")
    )
    return ChangedFile(
        path=path,
        status=status,  # type: ignore[arg-type]
        additions=additions,
        deletions=deletions,
        patch=patch,
        content_base=content_base,
        content_head=content_head,
        previous_path=previous_path,
    )


def _make_finding(
    *,
    finding_type: FindingType,
    file_path: str,
    line_start: int,
    line_end: int | None = None,
    review_id: UUID,
    installation_id: int,
) -> ReviewFinding:
    """Build a synthetic ReviewFinding pinned to file_path:line_start.

    Severity comes from `SEVERITY_POLICY[finding_type]` so the harness
    cannot accidentally encode a stale severity-vs-type mapping. The
    harness asserts (after building) that severity ∈ {MEDIUM, LOW, INFO}
    to satisfy the V1 eligibility gate.
    """
    if line_end is None:
        line_end = line_start
    severity = SEVERITY_POLICY[finding_type]
    if severity not in {FindingSeverity.MEDIUM, FindingSeverity.LOW, FindingSeverity.INFO}:
        raise ValueError(
            f"smoke harness rejects severity={severity!r} for {finding_type!r}; "
            f"V1 eligibility withholds CRITICAL/HIGH (hitl_required_node_absent), "
            f"which would produce a green run with zero comments posted (false-success bug)."
        )
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=review_id,
        installation_id=installation_id,
        finding_type=finding_type,
        severity=severity,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        title=f"smoke: {finding_type.value} at {file_path}:{line_start}",
        description=(
            "Synthetic finding from spikes/publish/smoke_publish.py. "
            "If you're seeing this on a real PR, the smoke harness was "
            "run with --apply against the smoke-test repo."
        ),
        evidence=f"line {line_start}",
        dimension=lookup_dimension(finding_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
    )


def _make_state(
    *,
    review_id: UUID,
    installation_id: int,
    owner: str,
    repo: str,
    pr_number: int,
    head_sha: str,
    base_sha: str,
    changed_files: tuple[ChangedFile, ...],
    findings: tuple[ReviewFinding, ...],
) -> ReviewState:
    """Build a `ReviewState` for the harness. Always `is_eval=True`."""
    from outrider.policy.canonical import compute_round_id
    from outrider.schemas.analysis_round import AnalysisRound

    pr_context = PRContext(
        installation_id=installation_id,
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        pr_title=f"smoke-test PR #{pr_number}",
        base_sha=base_sha,
        head_sha=head_sha,
        author="outrider-smoke-harness",
        total_additions=sum(cf.additions for cf in changed_files),
        total_deletions=sum(cf.deletions for cf in changed_files),
        changed_files=changed_files,
    )
    files_examined = tuple(cf.path for cf in changed_files)
    round_id = compute_round_id(
        pass_index=0,
        files_examined=files_examined,
        files_skipped=(),
        finding_content_hashes=tuple(f.content_hash for f in findings),
    )
    analysis_round = AnalysisRound(
        round_id=round_id,
        pass_index=0,
        findings=findings,
        files_examined=files_examined,
        files_skipped=(),
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
    )
    state = ReviewState(
        review_id=review_id,
        pr_context=pr_context,
        received_at=datetime.now(UTC),
        is_eval=True,  # hardcoded — see docs/testing.md eval isolation
        analysis_rounds=[analysis_round],
    )
    # State-is-pure-data check: round-trip through JSON. Catches embedded
    # clients/sessions/callbacks at harness boot, not deep inside publish().
    state.model_dump_json()
    return state


# ---------------------------------------------------------------------------
# Assertion bundles. Each returns (ok, reason) — operator-readable.
# ---------------------------------------------------------------------------


def _assert_first_publish(
    *,
    result: PublishResult,
    sink: _RecordingPublishEventSink,
    publisher: _StubPublisher,
    expected_eligible_count: int,
) -> list[tuple[bool, str]]:
    """Three-pillar success: result + audit emits + publisher call."""
    checks: list[tuple[bool, str]] = []
    checks.append(
        (
            result.outcome == "success",
            f"PublishResult.outcome == 'success' (got {result.outcome!r})",
        )
    )
    checks.append(
        (
            result.github_review_id is not None,
            f"PublishResult.github_review_id is not None (got {result.github_review_id})",
        )
    )
    checks.append(
        (
            result.comments_posted == expected_eligible_count,
            f"comments_posted == {expected_eligible_count} (got {result.comments_posted})",
        )
    )
    success_attempts = [a for a in sink.attempts if a.outcome is PublishAttemptOutcome.SUCCESS]
    checks.append(
        (
            len(success_attempts) == 1,
            f"exactly 1 PublishAttemptEvent(SUCCESS) (got {len(success_attempts)})",
        )
    )
    checks.append(
        (
            len(sink.results) == 1,
            f"exactly 1 PublishEvent (got {len(sink.results)})",
        )
    )
    eligible_emits = [e for e in sink.eligibility if e.eligibility is PublishEligibility.ELIGIBLE]
    checks.append(
        (
            len(eligible_emits) == expected_eligible_count,
            f"exactly {expected_eligible_count} PublishEligibilityEvent(ELIGIBLE) "
            f"(got {len(eligible_emits)})",
        )
    )
    checks.append(
        (
            len(publisher.create_calls) == 1,
            f"publisher.create_review called exactly once (got {len(publisher.create_calls)})",
        )
    )
    return checks


def _assert_idempotency(
    *,
    second_result: PublishResult,
    sink: _RecordingPublishEventSink,
    publisher: _StubPublisher,
) -> list[tuple[bool, str]]:
    """Second-invoke MUST short-circuit. No second POST. No second PublishEvent."""
    checks: list[tuple[bool, str]] = []
    checks.append(
        (
            second_result.outcome == "idempotently_skipped",
            f"second PublishResult.outcome == 'idempotently_skipped' "
            f"(got {second_result.outcome!r})",
        )
    )
    skipped_attempts = [
        a for a in sink.attempts if a.outcome is PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED
    ]
    checks.append(
        (
            len(skipped_attempts) == 1,
            f"exactly 1 PublishAttemptEvent(IDEMPOTENTLY_SKIPPED) (got {len(skipped_attempts)})",
        )
    )
    checks.append(
        (
            len(sink.results) == 1,
            f"PublishEvent count UNCHANGED at 1 after second invoke (got {len(sink.results)})",
        )
    )
    checks.append(
        (
            len(publisher.create_calls) == 1,
            f"publisher.create_review NOT called again "
            f"(total calls still 1; got {len(publisher.create_calls)})",
        )
    )
    return checks


def _phase_markers_after(sink: _RecordingPhaseEventSink, *, prior_count: int) -> list[str]:
    """Return the marker strings ('start'/'end') for events past `prior_count`.

    `ReviewPhaseEvent.marker` is `Literal["start", "end"]` per
    `audit/events.py:275` — a plain string, not an enum.
    """
    return [e.marker for e in sink.events[prior_count:]]


def _print_check_block(title: str, checks: Sequence[tuple[bool, str]]) -> bool:
    print(f"\n[{title}]")
    all_ok = True
    for ok, reason in checks:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {reason}")
        all_ok = all_ok and ok
    return all_ok


# ---------------------------------------------------------------------------
# Mock mode — runs anywhere; the publish node body runs against real
# schemas + stub publisher + recording sinks.
# ---------------------------------------------------------------------------


async def _run_mock_mode(args: argparse.Namespace) -> int:
    print("=== mock mode ===")
    print("  external deps: none")
    print("  asserting wiring + idempotency against in-memory sinks\n")

    review_id = uuid4()
    installation_id = 99_999  # synthetic; never reaches GitHub in mock mode
    head_sha = "0" * 40
    base_sha = "1" * 40

    # Fixture: an added-file patch (status='added' → content_base=None
    # is valid per ChangedFile's validator). The finding targets line 2
    # of head_content, which falls inside the added-file hunk.
    changed_file = _make_changed_file(
        path="src/example.py",
        content_head="def foo():\n    return 1\n\ndef bar():\n    return 2\n",
        patch=("@@ -0,0 +1,5 @@\n+def foo():\n+    return 1\n+\n+def bar():\n+    return 2\n"),
        status="added",
    )
    finding = _make_finding(
        finding_type=FindingType.MISSING_INPUT_VALIDATION,
        file_path="src/example.py",
        line_start=2,
        review_id=review_id,
        installation_id=installation_id,
    )
    state = _make_state(
        review_id=review_id,
        installation_id=installation_id,
        owner="ddamme05",
        repo="outrider-smoke-test",
        pr_number=args.pr,
        head_sha=head_sha,
        base_sha=base_sha,
        changed_files=(changed_file,),
        findings=(finding,),
    )

    publisher = _StubPublisher()
    phase_sink = _RecordingPhaseEventSink()
    publish_sink = _RecordingPublishEventSink()

    # First invoke — happy-path success.
    first = await publish(
        state,
        publisher=publisher,
        publish_event_sink=publish_sink,
        phase_event_sink=phase_sink,
        github_factory=_stub_github_factory,
    )

    first_checks = _assert_first_publish(
        result=first["publish_result"],
        sink=publish_sink,
        publisher=publisher,
        expected_eligible_count=1,
    )
    # Phase-events-bound-work: success path emits one start + one end.
    first_checks.append(
        (
            _phase_markers_after(phase_sink, prior_count=0) == ["start", "end"],
            f"first invoke phase events == [start, end] "
            f"(got {_phase_markers_after(phase_sink, prior_count=0)})",
        )
    )
    first_ok = _print_check_block("first invoke (success path)", first_checks)

    # Second invoke — simulate the intra-Outrider idempotency path. Set
    # the recording sink's `prior_publish_event` to what we just emitted
    # so the publish node's Step 4 lookup returns it. Convert the bare
    # assert to a check-block entry: under `python -O`, asserts are
    # compiled out, and a failure here should surface in the operator
    # output, not as a traceback.
    setup_check = (
        len(publish_sink.results) == 1,
        f"first invoke produced exactly 1 PublishEvent before idempotency "
        f"setup (got {len(publish_sink.results)})",
    )
    if not setup_check[0]:
        _print_check_block("idempotency setup", [setup_check])
        return 1
    publish_sink.prior_publish_event = publish_sink.results[0]
    prior_phase_count = len(phase_sink.events)

    second = await publish(
        state,
        publisher=publisher,
        publish_event_sink=publish_sink,
        phase_event_sink=phase_sink,
        github_factory=_stub_github_factory,
    )

    idem_checks = _assert_idempotency(
        second_result=second["publish_result"],
        sink=publish_sink,
        publisher=publisher,
    )
    # Phase-events-bound-work: idempotency path ALSO emits start + end
    # per spec (the phase brackets the work, even when the work is "no
    # POST happened").
    idem_checks.append(
        (
            _phase_markers_after(phase_sink, prior_count=prior_phase_count) == ["start", "end"],
            f"second invoke phase events == [start, end] "
            f"(got {_phase_markers_after(phase_sink, prior_count=prior_phase_count)})",
        )
    )
    idem_ok = _print_check_block("second invoke (idempotency path)", idem_checks)

    all_ok = first_ok and idem_ok
    print()
    print("=== mock mode result:", "PASS" if all_ok else "FAIL", "===")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Live mode — real GitHub + real Postgres. Hard-gated.
# ---------------------------------------------------------------------------


def _env_or_die(name: str, *, hint: str = "") -> str:
    value = os.environ.get(name)
    if not value:
        hint_text = f" ({hint})" if hint else ""
        raise SystemExit(f"required env var {name} is unset{hint_text}")
    return value


def _assert_test_database_url(url: str) -> None:
    """Refuse to run live mode against dev/prod Postgres.

    Parses the URL with `urlsplit` so a password containing `:5433/`
    can't spuriously pass the substring guard the integration-conftest
    version has. Requires port == 5433 (the `postgres-test` container
    per `docs/testing.md`) AND db name contains `outrider_test`.
    """
    # Strip the SQLAlchemy driver prefix so urlsplit parses port/path
    # correctly (`postgresql+psycopg://...` → `postgresql://...`).
    canonical = url.replace("+psycopg", "", 1).replace("+asyncpg", "", 1)
    parts = urlsplit(canonical)
    safe_target = f"{parts.hostname or '?'}:{parts.port or '?'}{parts.path}"
    if parts.port != 5433:
        raise SystemExit(
            f"refusing to run live mode: TEST_DATABASE_URL parsed port "
            f"{parts.port!r}, must be 5433 (the postgres-test container). "
            f"Target: {safe_target}"
        )
    db_name = parts.path.lstrip("/")
    if "outrider_test" not in db_name:
        raise SystemExit(
            f"refusing to run live mode: TEST_DATABASE_URL db name "
            f"{db_name!r} does not contain 'outrider_test'. Defense "
            f"against pointing the harness at dev/prod. Target: {safe_target}"
        )


def _assert_repo_allowlisted(owner: str, repo: str) -> None:
    if (owner, repo) not in _REPO_ALLOWLIST:
        raise SystemExit(
            f"refusing to run live mode against {owner}/{repo}; allowlist is "
            f"{sorted(_REPO_ALLOWLIST)!r}. Adding a target requires editing "
            f"spikes/publish/smoke_publish.py:_REPO_ALLOWLIST and routing the "
            f"change through PR review."
        )


async def _run_live_mode(args: argparse.Namespace) -> int:
    print("=== live mode ===")
    print(f"  target: {args.repo} PR #{args.pr}")
    print(f"  file: {args.file_path}:{args.line}")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │ LIVE MODE — will POST a real inline review comment to       │")
    print("  │ GitHub and write audit-event rows to the test Postgres.     │")
    print("  │ Per-run state recorded in spikes/publish/cleanup_manifest   │")
    print("  │ .jsonl (gitignored). Operator dismisses posted reviews      │")
    print("  │ manually; the publish path is append-only.                  │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    if "/" not in args.repo:
        raise SystemExit("--repo must be owner/name")
    owner, repo = args.repo.split("/", 1)
    _assert_repo_allowlisted(owner, repo)

    # Env gates BEFORE any imports that pull in pydantic-settings; if
    # the operator forgot a var, fail before constructing anything.
    # Validate all required env vars up-front. `GitHubAppSettings()` is
    # constructed bare (env-driven) per production parity with
    # `api/lifespan.py`; the explicit reads here are just so the
    # operator gets a clear error before pydantic-settings raises.
    _env_or_die("OUTRIDER_GITHUB_APP_ID")
    _env_or_die(
        "OUTRIDER_GITHUB_APP_PRIVATE_KEY",
        hint="full PEM content (BEGIN..END), not a path — matches lifespan.py",
    )
    _env_or_die(
        "OUTRIDER_GITHUB_WEBHOOK_SECRET",
        hint="required by GitHubAppSettings even though smoke harness doesn't receive webhooks",
    )
    installation_id = int(_env_or_die("OUTRIDER_SMOKE_INSTALLATION_ID"))
    database_url = _env_or_die("TEST_DATABASE_URL", hint="psycopg async URL, port 5433")
    _assert_test_database_url(database_url)

    print(f"  installation_id={installation_id}")
    print(f"  database: {database_url.rsplit('@', 1)[-1]}")

    # Imports deferred to live mode so mock mode runs without these
    # being importable (e.g., no sqlalchemy installed in a minimal env).
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from outrider.audit.config import RetentionSettings
    from outrider.audit.persister import AuditPersister
    from outrider.github.auth import make_installation_client_factory
    from outrider.github.config import GitHubAppSettings
    from outrider.github.fetch import fetch_file_content_at, list_pr_files
    from outrider.github.publisher import (
        GitHubKitPublisher,
        GitHubReviewValidationError,
        GitHubSecondaryRateLimitError,
    )

    # Bare-call construction matches production lifespan exactly; env
    # vars feed `GitHubAppSettings` via `env_prefix="OUTRIDER_GITHUB_"`.
    settings = GitHubAppSettings()
    github_factory = make_installation_client_factory(settings)
    gh = github_factory(installation_id)

    # Phase-1: list PR files; find ours; capture head_sha. Defend
    # against the PR being closed/merged — both have stale head SHAs
    # relative to what the operator likely intended.
    try:
        pr_resp = await gh.rest.pulls.async_get(owner, repo, args.pr)
    except Exception as exc:
        # Most common case here: 404 (App not installed on repo) or 403
        # (App installed but missing scope). Surface the likely cause
        # rather than a bare githubkit traceback.
        raise SystemExit(
            f"GET /repos/{owner}/{repo}/pulls/{args.pr} failed: "
            f"{exc.__class__.__name__}: {exc}. Most likely the App is not "
            f"installed on this repo OR the installation_id "
            f"({installation_id}) doesn't grant access to it. Check the "
            f"App's installation list at https://github.com/settings/installations."
        ) from exc
    pr = pr_resp.parsed_data
    if pr.state != "open":
        raise SystemExit(
            f"refusing to run live mode against PR #{args.pr} (state={pr.state!r}); "
            f"closed/merged PRs have stale or merge-anchor head SHAs that "
            f"likely don't match the diff the operator expected."
        )
    head_sha = pr.head.sha
    base_sha = pr.base.sha
    print(f"  PR head_sha={head_sha} base_sha={base_sha}")

    try:
        files = await list_pr_files(gh, owner=owner, repo=repo, pull_number=args.pr)
    except Exception as exc:
        raise SystemExit(
            f"GET /repos/{owner}/{repo}/pulls/{args.pr}/files failed: "
            f"{exc.__class__.__name__}: {exc}."
        ) from exc
    target = next((f for f in files if f.filename == args.file_path), None)
    if target is None:
        # Trim the available-list if huge — operator scans for their
        # filename, not the full inventory.
        available = sorted(f.filename for f in files)
        if len(available) > 12:
            available = available[:12] + [f"... ({len(files) - 12} more)"]
        raise SystemExit(
            f"--file-path {args.file_path!r} not in PR #{args.pr} diff "
            f"at head_sha={head_sha}; the file may have been renamed or "
            f"removed since the operator last set --file-path. Available: "
            f"{available}"
        )
    if not getattr(target, "patch", None):
        raise SystemExit(
            f"target file {args.file_path!r} has no patch field (likely too large "
            f"or binary). Pick a small text file change for the smoke."
        )

    # Phase-2: fetch head content via the production helper (path
    # validation included, per paths-validated-before-use).
    try:
        content_bytes = await fetch_file_content_at(
            gh, owner=owner, repo=repo, path=args.file_path, ref=head_sha
        )
    except Exception as exc:
        raise SystemExit(
            f"GET /repos/{owner}/{repo}/contents/{args.file_path}?ref={head_sha} "
            f"failed: {exc.__class__.__name__}: {exc}."
        ) from exc
    if content_bytes is None:
        raise SystemExit(
            f"fetch_file_content_at returned None for {args.file_path!r} "
            f"(oversize, non-file, or symlink). Pick another path."
        )
    content_head = content_bytes.decode("utf-8")

    # Line bounds check: `--line` must be within head_content. The
    # publish-node coordinate translator catches past-EOF eventually
    # (BYTE_OFFSET_INVALID → DASHBOARD_ONLY routing), but that yields
    # a confusing "no inline comment posted" result instead of a clear
    # "you pointed at a non-existent line" operator error.
    head_line_count = len(content_head.splitlines())
    if not 1 <= args.line <= head_line_count:
        raise SystemExit(
            f"--line {args.line} out of range for {args.file_path!r} at "
            f"head_sha={head_sha} ({head_line_count} lines). Pick a line "
            f"in [1, {head_line_count}]."
        )

    # Base-content fetch per status. `ChangedFile`'s validator enforces:
    # `modified`/`renamed` need both base + head; `added` only head;
    # `removed` only base (and can't be inline-commented — skip).
    content_base: str | None = None
    previous_path: str | None = None
    if target.status in {"modified", "renamed"}:
        # Vendor-quirk normalization mirroring `agent/nodes/intake.py:480`:
        # GitHubKit returns `previous_filename=""` for non-renamed files,
        # AND can plausibly return `""` for renamed if a future drift
        # hits. `or None` collapses both None and `""` to None so the
        # rename-without-previous-filename case fails loud with a
        # helpful message instead of an opaque ValidationError.
        raw_previous_filename = getattr(target, "previous_filename", None) or None
        if target.status == "renamed":
            if raw_previous_filename is None:
                raise SystemExit(
                    f"GitHub returned status='renamed' for {args.file_path!r} "
                    f"without previous_filename — likely a GitHubKit shape "
                    f"drift; production intake.py:642 has the matching guard."
                )
            base_path = raw_previous_filename
        else:
            base_path = args.file_path
        try:
            base_bytes = await fetch_file_content_at(
                gh, owner=owner, repo=repo, path=base_path, ref=base_sha
            )
        except Exception as exc:
            raise SystemExit(
                f"GET /repos/{owner}/{repo}/contents/{base_path}?ref={base_sha} "
                f"(base content for {target.status}) failed: "
                f"{exc.__class__.__name__}: {exc}."
            ) from exc
        if base_bytes is None:
            raise SystemExit(
                f"fetch_file_content_at returned None for base path "
                f"{base_path!r} at ref={base_sha} (oversize/non-file/symlink). "
                f"Pick another file."
            )
        content_base = base_bytes.decode("utf-8")
        if target.status == "renamed":
            previous_path = base_path
    elif target.status == "removed":
        raise SystemExit(
            f"target file {args.file_path!r} status is 'removed'; the publish "
            f"node correctly routes removed-file findings to DASHBOARD_ONLY "
            f"(HEAD_CONTENT_UNAVAILABLE) — which means zero inline comments "
            f"post and the smoke can't exercise the happy path. Pick an "
            f"added/modified/renamed file instead."
        )

    changed_file = _make_changed_file(
        path=args.file_path,
        content_head=content_head,
        content_base=content_base,
        patch=target.patch,
        status=target.status,
        previous_path=previous_path,
    )

    review_id = uuid4()
    finding = _make_finding(
        finding_type=FindingType.MISSING_INPUT_VALIDATION,
        file_path=args.file_path,
        line_start=args.line,
        review_id=review_id,
        installation_id=installation_id,
    )
    state = _make_state(
        review_id=review_id,
        installation_id=installation_id,
        owner=owner,
        repo=repo,
        pr_number=args.pr,
        head_sha=head_sha,
        base_sha=base_sha,
        changed_files=(changed_file,),
        findings=(finding,),
    )
    print(f"  review_id={review_id}")

    # Wire real persister + publisher.
    async with AsyncExitStack() as stack:
        engine = create_async_engine(database_url, hide_parameters=True)
        stack.push_async_callback(engine.dispose)

        # Postgres connectivity probe BEFORE the first publish call so
        # "docker compose up -d postgres-test missing" surfaces with a
        # clear error message instead of crashing deep inside emit_phase.
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:
            raise SystemExit(
                f"Postgres connectivity probe failed against "
                f"{database_url.rsplit('@', 1)[-1]}: "
                f"{exc.__class__.__name__}: {exc}. Run "
                f"`docker compose up -d postgres-test` and re-try."
            ) from exc

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        persister = AuditPersister(
            session_factory=session_factory,
            retention_settings=RetentionSettings(),
        )
        publisher = GitHubKitPublisher()
        phase_sink = persister  # AuditPersister satisfies PhaseEventSink too

        # Manifest path is anchored to the harness file's directory, NOT
        # CWD — operator can run from anywhere and still find the record
        # under spikes/publish/.
        manifest_path = Path(__file__).parent / "cleanup_manifest.jsonl"

        print("\nfirst invoke (real POST):")
        try:
            first = await publish(
                state,
                publisher=publisher,
                publish_event_sink=persister,
                phase_event_sink=phase_sink,
                github_factory=github_factory,
            )
        except GitHubReviewValidationError as exc:
            raise SystemExit(
                f"GitHub rejected the publish request as a validation "
                f"failure (status={exc.status_code}). Common cause: the "
                f"--file-path / --line combination doesn't anchor to a "
                f"reviewable diff line at head_sha={head_sha}. "
                f"Inspect publisher.body_text for details."
            ) from exc
        except GitHubSecondaryRateLimitError as exc:
            raise SystemExit(
                f"GitHub returned a secondary rate limit (status={exc.status_code}). "
                f"Wait several minutes and re-run; do NOT re-run in a loop "
                f"(rate-limit windows compound)."
            ) from exc
        first_result: PublishResult = first["publish_result"]

        # Manifest append happens IMMEDIATELY after publish returns,
        # BEFORE any print. Closes the Ctrl-C race between GitHub POST
        # success and the record write — even if the operator hits Ctrl-C
        # before the pillar prints, the manifest entry already landed.
        if first_result.github_review_id is not None:
            _append_manifest(
                manifest_path,
                review_id=review_id,
                github_review_id=first_result.github_review_id,
                owner=owner,
                repo=repo,
                pr_number=args.pr,
            )

        print(
            f"  outcome={first_result.outcome!r} "
            f"github_review_id={first_result.github_review_id} "
            f"comments_posted={first_result.comments_posted}"
        )

        # Three-pillar check pillar 1: result shape.
        ok_result = (
            first_result.outcome == "success"
            and first_result.github_review_id is not None
            and first_result.comments_posted == 1
        )
        # Pillar 2: GitHub re-query for the body-marker. Confirms the
        # comment is visible via the same matcher the publisher uses
        # for idempotency lookups on subsequent webhook deliveries.
        body_marker = _BODY_MARKER_TEMPLATE.format(review_id=str(review_id))
        existing = await publisher.find_existing_review_on_head_sha(
            gh=gh,
            owner=owner,
            repo=repo,
            pull_number=args.pr,
            head_sha=head_sha,
            body_marker=body_marker,
        )
        ok_github = existing == first_result.github_review_id

        # Pillar 3 (audit-row counts) is DEFERRED to a future extension
        # because querying `audit_events` requires a separate
        # AsyncSession scope; the current harness only proves the
        # persister did not raise during emit (which Pillar 1's
        # success-outcome implicitly carries). Renamed [SKIP] so the
        # operator isn't misled into thinking we actually checked the
        # row count.

        print(
            "\n  [OK]   pillar 1: result shape"
            if ok_result
            else "\n  [FAIL] pillar 1: result shape"
        )
        print(
            f"  [{'OK' if ok_github else 'FAIL':4s}] pillar 2: GitHub re-query "
            f"(body_marker found: {existing})"
        )
        print(
            "  [SKIP] pillar 3: audit-row count "
            "(deferred — persister did not raise; no row-count verification yet)"
        )

        # Second invoke — must hit intra-Outrider idempotency.
        print("\nsecond invoke (idempotency check):")
        second = await publish(
            state,
            publisher=publisher,
            publish_event_sink=persister,
            phase_event_sink=phase_sink,
            github_factory=github_factory,
        )
        second_result: PublishResult = second["publish_result"]
        print(f"  outcome={second_result.outcome!r}")
        ok_idem = second_result.outcome == "idempotently_skipped"
        print(
            f"  [{'OK' if ok_idem else 'FAIL':4s}] idempotency "
            f"(expected outcome='idempotently_skipped'; got {second_result.outcome!r})"
        )

        # Pillar 3 (audit) is [SKIP] not [FAIL] — exclude from all_ok.
        all_ok = ok_result and ok_github and ok_idem
        print()
        print("=== live mode result:", "PASS" if all_ok else "FAIL", "===")
        print(f"  cleanup_manifest: {manifest_path}")
        return 0 if all_ok else 1


def _append_manifest(
    path: Path,
    *,
    review_id: UUID,
    github_review_id: int,
    owner: str,
    repo: str,
    pr_number: int,
) -> None:
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "review_id": str(review_id),
        "github_review_id": github_review_id,
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spikes.publish.smoke_publish",
        description="Smoke-test the V1 publish node end-to-end.",
        epilog=(
            "LIVE mode (--apply) requires these env vars set:\n"
            "  OUTRIDER_GITHUB_APP_ID, OUTRIDER_GITHUB_APP_PRIVATE_KEY,\n"
            "  OUTRIDER_GITHUB_WEBHOOK_SECRET, OUTRIDER_SMOKE_INSTALLATION_ID,\n"
            "  TEST_DATABASE_URL (port 5433, db name contains outrider_test).\n"
            "See spikes/publish/README.md for the full prerequisite list."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run in LIVE mode: real GitHub POST + real Postgres write. "
        "Requires env vars (OUTRIDER_GITHUB_*, OUTRIDER_SMOKE_INSTALLATION_ID, "
        "TEST_DATABASE_URL) and the postgres-test container running.",
    )
    parser.add_argument(
        "--repo",
        default="ddamme05/outrider-smoke-test",
        help="owner/name; allowlisted to ddamme05/outrider-smoke-test only.",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=2,
        help="PR number on --repo. Default 2 (the Q6 sandbox PR).",
    )
    parser.add_argument(
        "--file-path",
        default="README.md",
        help="Repo-relative path to anchor the smoke finding (must be in PR diff).",
    )
    parser.add_argument(
        "--line",
        type=int,
        default=20,
        help="1-indexed source line in --file-path on the head SHA.",
    )
    return parser.parse_args(argv)


async def amain(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.apply:
        return await _run_live_mode(args)
    return await _run_mock_mode(args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # KeyboardInterrupt is NOT suppressed — silently swallowing Ctrl-C
    # between a GitHub POST and the cleanup_manifest append would leave
    # the operator with no record of the just-posted review. asyncio.run
    # raises KeyboardInterrupt as-is on Ctrl-C; let it propagate so the
    # operator sees the traceback and the exit code is non-zero.
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
