"""End-to-end smoke test: real graph run through publish, then replay-equivalent.

It drives the REAL compiled graph (built from the full seven-node topology:
intake -> triage -> analyze <-> trace -> synthesize -> hitl -> publish) with
the REAL `AuditPersister` / `ReviewStatusPersister` / `AnomalyPersister` writing
to a REAL Postgres (`migrated_db`), and fakes only at the two network
boundaries -- a scripted LLM provider (no Anthropic call) and a fake GitHub
(stub fetch client + recording publisher). A synthetic one-finding PR runs all
the way THROUGH publish (the finding is MEDIUM, so the HITL gate does not
interrupt; it lands on a changed diff line, so publish actually posts an inline
comment instead of short-circuiting). The capstone then asserts the run
produced a faithfully replayable audit stream:
`AuditReplayer.assert_replay_equivalent(review_id)` passes and reconstruct
classifies the review FULL.

This is the happy-path slice, NOT a literal exercise of all seven nodes. The
nodes that actually run are intake, triage, analyze, synthesize, hitl, publish
(asserted via phase coverage below). Two nodes are present in the compiled
graph but NOT exercised by this fixture:
  - **trace does not run** -- the analyze response carries no `trace_candidates`,
    so `_analyze_router` skips the analyze<->trace loop. (A trace-exercising
    end-to-end path is separate scope.)
  - **hitl runs as a pass-through** -- the MEDIUM finding is sub-HIGH, so the
    gate emits its phase bracket but does not `interrupt()`. The HITL
    interrupt/resume path is covered by the hitl_resume scenario, not here.

Why this complements `test_full_mode_through_production_persister`: that test
drives the persister's methods directly; this one drives them THROUGH the graph,
so it proves the node->sink wiring and the graph-driven audit stream, not just
the persister in isolation.

One deliberate seam:
  - The scripted provider does NOT emit `LLMCallEvent` rows -- LLM-call
    persistence lives inside `AnthropicProvider.complete()` (covered
    separately), not in a graph sink. So the audit stream carries zero LLM
    exchanges here and `llm_exchanges` is vacuously empty; FULL mode does not
    require LLM content.

The `findings` CONTENT row is written by the production writer during the
graph's analyze node (`emit_finding` co-inserts the FindingEvent audit row and
the findings content row in one transaction, FUP-111), so the run reconstructs
FULL with finding content -- no raw-SQL content seed.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from outrider.agent.graph import build_graph
from outrider.agent.nodes.hitl_config import HITLConfig
from outrider.anomaly.persister import AnomalyPersister
from outrider.audit.config import RetentionSettings
from outrider.audit.events import PublishEvent
from outrider.audit.persister import AuditPersister
from outrider.audit.replay import AuditReplayer, ReplayMode
from outrider.db.review_status_persister import ReviewStatusPersister
from outrider.llm.config import ModelConfig
from outrider.schemas import GitHubReviewCreated
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_state import ReviewState

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from outrider.github import InstallationGitHubClient
    from outrider.llm.base import LLMRequest, LLMResponse
    from outrider.schemas import InlineComment

pytestmark = pytest.mark.asyncio

_INSTALLATION_ID = 12345
_OWNER = "acme"
_REPO = "widget"
_PULL_NUMBER = 7
_BASE_SHA = "a" * 40
_HEAD_SHA = "b" * 40

# A synthetic file that ADDS a whole new function in the diff, so every line of
# the new function is a changed (added) line. The finding lands on the new
# function's body line -- guaranteed to be in the changed region, so publish
# routes it INLINE (not REVIEW_BODY / DASHBOARD_ONLY).
_FILE_PATH = "src/handler.py"
_BASE_CONTENT = "def existing():\n    return 1\n"
_HEAD_CONTENT = (
    "def existing():\n    return 1\n\ndef vulnerable(user_input):\n    return user_input\n"
)
_PATCH = (
    f"--- a/{_FILE_PATH}\n"
    f"+++ b/{_FILE_PATH}\n"
    "@@ -1,2 +1,5 @@\n"
    " def existing():\n"
    "     return 1\n"
    "+\n"
    "+def vulnerable(user_input):\n"
    "+    return user_input\n"
)

# Byte span of "    return user_input" within _HEAD_CONTENT -- the finding target.
_FINDING_BYTE_START = _HEAD_CONTENT.index("    return user_input")
_FINDING_BYTE_END = _FINDING_BYTE_START + len("    return user_input")


# ---------------------------------------------------------------------------
# Scripted LLM provider (routes canned responses by node_id; never calls a SDK)
# ---------------------------------------------------------------------------


class _ScriptedLLMProvider:
    """Returns canned responses keyed by `request.node_id`.

    Mirrors `tests/integration/test_analyze_graph_wiring.py`'s mock. Does NOT
    persist `LLMCallEvent`s -- that lives in `AnthropicProvider`, not a graph
    sink -- so the audit stream here has no LLM exchanges (fine for FULL mode).
    """

    def __init__(self, *, triage_response: str, analyze_response: str) -> None:
        self.triage_response = triage_response
        self.analyze_response = analyze_response
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        from outrider.llm.base import LLMResponse

        self.calls.append(request)
        if request.node_id == "triage":
            text_out = self.triage_response
        elif request.node_id == "analyze":
            text_out = self.analyze_response
        elif request.node_id == "synthesize":
            text_out = "Smoke test: one input-validation finding on the new function."
        else:
            msg = f"unexpected node_id in scripted provider: {request.node_id!r}"
            raise AssertionError(msg)
        return LLMResponse(
            text=text_out,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=42,
        )


# ---------------------------------------------------------------------------
# Fake GitHub: stub fetch client (intake) + recording publisher (publish)
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
    encoding: str
    content: str


@dataclass
class _StubResponse:
    parsed_data: Any


class _StubReposAPI:
    async def async_get_content(
        self, owner: str, repo: str, path: str, *, ref: str
    ) -> _StubResponse:
        content_bytes = _BASE_CONTENT.encode() if ref == _BASE_SHA else _HEAD_CONTENT.encode()
        # Single-line base64 (payload is <60 chars, so real GitHub wouldn't wrap
        # it either). If _HEAD_CONTENT grows past 60 base64 chars, switch to
        # base64.encodebytes to match GitHub's newline-wrapped contents-API shape
        # (see feedback_test_stubs_match_wire_format).
        return _StubResponse(
            parsed_data=_StubContentFile(
                encoding="base64",
                content=base64.b64encode(content_bytes).decode("ascii"),
            )
        )


class _StubPullsAPI:
    async def async_list_files(
        self, owner: str, repo: str, pull_number: int, **kwargs: Any
    ) -> _StubResponse:
        return _StubResponse(
            parsed_data=[
                _StubFileMeta(
                    filename=_FILE_PATH,
                    status="modified",
                    additions=3,
                    deletions=0,
                    patch=_PATCH,
                )
            ]
        )


class _StubRestAPI:
    def __init__(self) -> None:
        self.repos = _StubReposAPI()
        self.pulls = _StubPullsAPI()


class _StubGitHub:
    def __init__(self) -> None:
        self.rest = _StubRestAPI()


def _stub_github_factory(installation_id: int) -> Any:
    assert installation_id == _INSTALLATION_ID, f"unexpected installation_id {installation_id}"
    return _StubGitHub()


@dataclass
class _RecordingPublisher:
    """Fake `GitHubPublisher`: records create_review, no prior review exists."""

    create_review_calls: list[dict[str, Any]] = field(default_factory=list)

    async def create_review(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        review_status: str,
        body_marker: str,
        comments: tuple[InlineComment, ...],
    ) -> GitHubReviewCreated:
        self.create_review_calls.append(
            {
                "review_status": review_status,
                "comments": comments,
                "head_sha": head_sha,
                "pull_number": pull_number,
            }
        )
        return GitHubReviewCreated(github_review_id=999, comments_posted=len(comments))

    async def find_existing_review_on_head_sha(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        body_marker: str,
    ) -> int | None:
        return None


class _StubImportPathResolver:
    """Satisfies `build_graph`'s ImportPathResolver Protocol gate.

    Trace does not run in this fixture (the analyze response carries no
    `trace_candidates`), so `resolve_candidate_paths` is never actually called.
    """

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        return []


# ---------------------------------------------------------------------------
# Canned LLM payloads
# ---------------------------------------------------------------------------


def _triage_response() -> str:
    return json.dumps(
        {
            "file_tiers": {_FILE_PATH: "deep"},
            "overall_risk": "medium",
            "relevant_dimensions": ["security"],
            "reasoning": "smoke: deep-review the changed handler.",
        }
    )


def _analyze_response() -> str:
    # MEDIUM finding (missing_input_validation) on the added function body line.
    return json.dumps(
        {
            "findings": [
                {
                    "finding_type": "missing_input_validation",
                    "evidence_tier": "judged",
                    "query_match_id": None,
                    "trace_path": None,
                    "title": "Unvalidated user input returned directly",
                    "description": "vulnerable() returns user_input without validation.",
                    "evidence": "    return user_input",
                    "span": {"byte_start": _FINDING_BYTE_START, "byte_end": _FINDING_BYTE_END},
                    "trace_candidates": [],
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# Seed helpers (FK-ordered raw SQL; is_eval omitted => server default false)
# ---------------------------------------------------------------------------


async def _seed_installation(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations (installation_id, app_slug, account_id, "
                "account_login, account_type, permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb) "
                "ON CONFLICT (installation_id) DO NOTHING"
            ),
            {"id": _INSTALLATION_ID},
        )


async def _seed_review(engine: AsyncEngine, review_id: UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO reviews (id, installation_id, repo_id, pr_number, head_sha, "
                "status, files_examined, files_traced_beyond_diff, llm_calls_made, "
                "total_input_tokens, total_output_tokens, total_cost_usd, wall_clock_seconds, "
                "retention_expires_at) VALUES (:id, :iid, 100, :pr, :sha, 'running', "
                "0, 0, 0, 0, 0, 0, 0, NOW() + INTERVAL '180 days')"
            ),
            {"id": review_id, "iid": _INSTALLATION_ID, "pr": _PULL_NUMBER, "sha": _HEAD_SHA},
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine(migrated_db: str) -> AsyncGenerator[AsyncEngine]:
    eng = create_async_engine(migrated_db, poolclass=NullPool)
    try:
        yield eng
    finally:
        await eng.dispose()


def _seed_state(review_id: UUID) -> ReviewState:
    return ReviewState(
        review_id=review_id,
        received_at=datetime.now(UTC),
        pr_context=PRContext(
            installation_id=_INSTALLATION_ID,
            owner=_OWNER,
            repo=_REPO,
            pr_number=_PULL_NUMBER,
            base_sha=_BASE_SHA,
            head_sha=_HEAD_SHA,
            pr_title="Add vulnerable handler",
            pr_body=None,
            author="someone",
            total_additions=3,
            total_deletions=0,
            changed_files=(
                ChangedFile(
                    path=_FILE_PATH,
                    status="modified",
                    additions=3,
                    deletions=0,
                    patch=_PATCH,
                    content_base=_BASE_CONTENT,
                    content_head=_HEAD_CONTENT,
                    previous_path=None,
                    language="python",
                ),
            ),
        ),
        is_eval=False,
    )


# ---------------------------------------------------------------------------
# The smoke test
# ---------------------------------------------------------------------------


async def test_full_review_reaches_publish_and_replays_equivalent(engine: AsyncEngine) -> None:
    review_id = uuid4()
    await _seed_installation(engine)
    await _seed_review(engine, review_id)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    persister = AuditPersister(
        session_factory=session_factory, retention_settings=RetentionSettings()
    )
    review_status_sink = ReviewStatusPersister(session_factory=session_factory)
    anomaly_sink = AnomalyPersister(session_factory=session_factory)
    publisher = _RecordingPublisher()
    provider = _ScriptedLLMProvider(
        triage_response=_triage_response(), analyze_response=_analyze_response()
    )

    graph = build_graph(
        db_factory=session_factory,
        github_factory=_stub_github_factory,
        provider=provider,
        model_config=ModelConfig(),
        phase_event_sink=persister,
        file_examination_sink=persister,
        analyze_event_sink=persister,
        publish_event_sink=persister,
        trace_sink=persister,
        hitl_event_sink=persister,
        synthesize_event_sink=persister,
        review_status_sink=review_status_sink,
        anomaly_sink=anomaly_sink,
        hitl_config=HITLConfig(),
        checkpointer=InMemorySaver(),
        publisher=publisher,
        import_path_resolver=_StubImportPathResolver(),
    )

    result = await graph.ainvoke(
        _seed_state(review_id),
        config={"configurable": {"thread_id": str(review_id)}},
    )
    assert result is not None

    # publish actually posted: exactly one create_review with >=1 inline comment.
    assert len(publisher.create_review_calls) == 1, (
        f"expected exactly one create_review call, got {len(publisher.create_review_calls)} "
        "(if zero, the finding did not route inline and publish short-circuited)"
    )
    assert len(publisher.create_review_calls[0]["comments"]) >= 1
    # V1 contract: publish always posts review_status=COMMENT (publish.py hardcodes
    # it; severity-derived status is a future enhancement). Pinning it trips this
    # test if that contract changes silently.
    assert publisher.create_review_calls[0]["review_status"] == "COMMENT"

    # Terminal state (state side, corroborating the publisher-call assertion):
    # the graph's PublishResult records a successful post. `outcome` is the plain
    # string the schema uses (no enum) — `success` for a posted review.
    publish_result = result["publish_result"]
    assert publish_result is not None
    assert publish_result.outcome == "success"
    assert publish_result.github_review_id == 999
    assert publish_result.comments_posted == 1

    # review flipped running -> completed.
    async with engine.begin() as conn:
        status = (
            await conn.execute(text("SELECT status FROM reviews WHERE id = :id"), {"id": review_id})
        ).scalar_one()
    assert status == "completed"

    # Reconstruct: the finding audit row AND its content row both landed via the
    # production writer (emit_finding co-inserts them in one transaction inside the
    # graph's analyze node) — no raw-SQL content seed needed (FUP-111 closed).
    replayer = AuditReplayer(session_factory=session_factory)
    pre = await replayer.reconstruct(review_id)
    assert len(pre.findings) == 1, (
        f"expected one FindingEvent in the stream, got {len(pre.findings)}"
    )
    assert pre.findings[0].content is not None, (
        "the production findings-content writer should have co-inserted the content "
        "row during the graph's analyze node, so it reconstructs FULL with content"
    )

    # Phase pairs for the nodes that ran (publish ran -- no HITL interrupt).
    started = {p.node_id for p in pre.phases}
    assert {"intake", "triage", "analyze", "synthesize", "hitl", "publish"} <= started, (
        f"expected full-graph node coverage through publish, got {sorted(started)}"
    )

    # The publish leg is auditable: a PublishEvent landed in the stream (NOT just
    # the publish phase-bracket). Phase markers come from a different sink path
    # (`emit_phase`) than the publish events (`emit_publish_result`), so without
    # this check the publish-side audit wiring could silently drop every event
    # and the phase-coverage assertion above would still pass.
    publish_events = [e for e in pre.events if isinstance(e, PublishEvent)]
    assert len(publish_events) == 1, (
        f"expected exactly one PublishEvent in the audit stream, got {len(publish_events)}"
    )
    assert publish_events[0].review_status == "COMMENT"
    assert publish_events[0].comments_posted == 1

    # Capstone: a graph-driven run replays faithfully in FULL mode.
    post = await replayer.reconstruct(review_id)
    assert post.mode == ReplayMode.FULL
    # is_eval (False here) propagated graph-wide to the reconstructed read model.
    assert post.is_eval is False
    await replayer.assert_replay_equivalent(review_id)  # no raise
