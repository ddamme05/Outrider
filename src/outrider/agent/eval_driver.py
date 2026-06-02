# Fixture-driven eval graph driver: run_review + the three src-local replay adapters.
# See specs/2026-06-01-eval-graph-driver.md (resolution A). LLM-provider boundary #8 +
# input boundary #5: the adapters implement the LLMProvider / GitHub / GitHubPublisher
# Protocols WITHOUT importing anthropic/githubkit — they replay fixture data only.
"""The eval graph driver — drive the real 7-node graph from a JSON fixture.

`run_review(fixture_path)` is the shim every non-structural eval scenario
imports (`from outrider.agent import run_review`). It generalizes the
CI-gated wiring in `tests/integration/test_e2e_smoke.py`: it builds the real
compiled graph with real audit persisters against an ephemeral
`postgres-test` database, fakes only the two network boundaries (a scripted
`LLMProvider` and a fake GitHub read client + capturing publisher), runs the
graph to completion, and returns an `EvalRunResult`.

**Why this lives in `src/`** (resolution A): the scenarios call
`run_review("…json")` with only a path — no injection seam — so the shim
must self-construct its dependencies, and the import contract
(`outrider.agent.run_review`) forces it to be `src`-reachable. The three
adapters it constructs (`_FixtureScriptedProvider`, `_FixtureGitHubClient`,
`_CapturingPublisher`) are therefore `src`-local, but they are a
*data-driven replay engine*, not hand-coded mock scaffolding: each reads its
responses straight from the fixture and imports no vendor SDK. Bespoke
assertion-oriented doubles stay under `tests/`.

**Fail-closed.** `run_review` calls `require_eval_mode()` first (OUTRIDER_IS_EVAL=1)
and the ephemeral-DB helper runs the port-5433/"test" URL guard before any
DDL — this shim is `src`-reachable and must refuse to touch a non-test DB.

**Scripted-vs-real caveat** (see the spec's Non-goals): a scripted LLM proves
*pipeline handling of a given response*, NOT that the real model returns
in-scope coordinates. That is a separate live-eval feature.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, ConfigDict
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
from outrider.audit.persister import AuditPersister
from outrider.db.review_status_persister import ReviewStatusPersister
from outrider.eval_support import (
    assert_no_is_eval_violations,
    ephemeral_database,
    require_eval_mode,
    run_alembic_upgrade_head,
)
from outrider.llm.config import ModelConfig
from outrider.schemas.hitl import HITLRequest
from outrider.schemas.pr_context import PRContext
from outrider.schemas.publish import GitHubReviewCreated
from outrider.schemas.review_state import ReviewState

if TYPE_CHECKING:
    from pathlib import Path

    from outrider.github import InstallationGitHubClient
    from outrider.llm.base import LLMRequest, LLMResponse
    from outrider.schemas.publish import InlineComment
    from outrider.schemas.review_finding import ReviewFinding
    from outrider.schemas.trace_decision import TraceDecision

# reviews.repo_id is an opaque int here; eval scenarios never assert on it.
_EVAL_REPO_ID = 100
# Env var naming the base postgres-test URL (port 5433); the ephemeral DB is
# carved off it per run.
_TEST_DB_URL_ENV_VAR = "TEST_DATABASE_URL"


class EvalDriverError(RuntimeError):
    """A fixture or scripted-response problem made the run unrunnable.

    Distinct from production `LLMProviderError` / `BuildGraphError`: this is a
    test-harness misconfiguration (missing scripted response, unset
    TEST_DATABASE_URL, malformed fixture), surfaced loudly so a scenario fails
    with a clear cause rather than a confusing downstream error.
    """


# ---------------------------------------------------------------------------
# Fixture schema
# ---------------------------------------------------------------------------


class EvalFixtureFile(BaseModel):
    """One changed file in the PR fixture. `content_*` follow GitHub status
    semantics: `added` → head only, `removed` → base only, `modified` → both,
    `renamed` → both + `previous_path`."""

    model_config = ConfigDict(extra="forbid")

    path: str
    status: Literal["added", "removed", "modified", "renamed"]
    additions: int
    deletions: int
    patch: str | None = None
    previous_path: str | None = None
    content_base: str | None = None
    content_head: str | None = None


class EvalFixture(BaseModel):
    """A complete PR fixture: identity + changed files + scripted LLM responses.

    `llm_responses` maps `node_id` → the ordered list of raw response strings
    that node's calls receive (index 0 = first call). The strings are the
    exact text the real node parser consumes (triage tiers JSON, the analyze
    findings JSON, synthesize summary prose) — the shape proven by
    `tests/integration/test_e2e_smoke.py`.
    """

    model_config = ConfigDict(extra="forbid")

    installation_id: int
    owner: str
    repo: str
    pr_number: int
    base_sha: str
    head_sha: str
    pr_title: str
    pr_body: str | None = None
    author: str
    total_additions: int
    total_deletions: int
    files: tuple[EvalFixtureFile, ...]
    llm_responses: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRunResult:
    """The terminal product of a driven review.

    Iterable + len-able over `.findings` so finding-scenarios can write
    `findings = run_review(...)` / `[f for f in findings ...]` / `len(...)`
    verbatim; also exposes `.findings` / `.trace_decisions` / `.review_id` /
    `.published_comments` so trace-accuracy + FUP-137 scenarios can reach
    graph-state and the captured publish (the contract is a result object,
    not a bare tuple — see the spec's return-type fork).

    **HITL gating is a legitimate terminal state, not a failure.** `run_review`
    is a single-pass driver: when a CRITICAL/HIGH finding trips the HITL gate,
    the graph `interrupt()`s and this driver STOPS there — it does NOT
    auto-approve (that would auto-publish a gated finding, defeating trust
    boundary #6; resume semantics are the separate, deferred
    `run_review_with_resume` shim's job). On a gated run:
      - `.findings` is still populated (synthesize ran before hitl);
      - `.published_comments` is `()` (publish never ran — the gate held);
      - `.hitl_gated` is `True` and `.hitl_request` carries the gate payload.
    So `published_comments == ()` with `hitl_gated=True` means "gated", which a
    scenario distinguishes from "no findings" via the flag.
    """

    review_id: UUID
    findings: tuple[ReviewFinding, ...]
    trace_decisions: tuple[TraceDecision, ...]
    published_comments: tuple[InlineComment, ...]
    hitl_gated: bool
    hitl_request: HITLRequest | None = None

    def __iter__(self) -> Any:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)


# ---------------------------------------------------------------------------
# Adapter 1 — scripted LLM provider (LLMProvider Protocol; no `anthropic`)
# ---------------------------------------------------------------------------


class _FixtureScriptedProvider:
    """Returns canned responses keyed by `(request.node_id, call-index)`.

    Implements the `LLMProvider` Protocol; imports no SDK. Token counts are
    fixed sentinels (cost/latency aren't asserted by scenarios). Raises
    `EvalDriverError` loudly when a node makes more calls than the fixture
    scripts, so a scenario fails with a clear cause.
    """

    def __init__(self, responses: dict[str, list[str]]) -> None:
        self._responses = responses
        self._counts: dict[str, int] = {}

    async def complete(self, request: LLMRequest) -> LLMResponse:
        from outrider.llm.base import LLMResponse

        node = request.node_id
        idx = self._counts.get(node, 0)
        try:
            text_out = self._responses[node][idx]
        except (KeyError, IndexError) as exc:
            raise EvalDriverError(
                f"no scripted LLM response for node_id={node!r} call-index={idx} "
                f"(fixture scripts {len(self._responses.get(node, []))} call(s) for "
                f"this node). Add the response to the fixture's llm_responses."
            ) from exc
        self._counts[node] = idx + 1
        return LLMResponse(
            text=text_out,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=1,
        )


# ---------------------------------------------------------------------------
# Adapter 2 — fake GitHub read client (intake's two-phase fetch; no `githubkit`)
# ---------------------------------------------------------------------------


@dataclass
class _FixtureFileMeta:
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None
    previous_filename: str | None


@dataclass
class _FixtureContentFile:
    encoding: str
    content: str


@dataclass
class _FixtureResponse:
    parsed_data: Any


class _FixtureReposAPI:
    """Serves `async_get_content` from fixture content, keyed by (path, ref).

    Content is base64 with newline wrapping (`base64.encodebytes`) to match
    GitHub's real contents-API wire shape — intake strips `\\n`/`\\r` then
    decodes with `validate=True` (see `github/fetch.py`), so an unwrapped stub
    would not exercise the strip path real GitHub triggers
    (`feedback_test_stubs_match_wire_format`).
    """

    def __init__(self, content_by_path_ref: dict[tuple[str, str], str]) -> None:
        self._content = content_by_path_ref

    async def async_get_content(
        self, owner: str, repo: str, path: str, *, ref: str
    ) -> _FixtureResponse:
        try:
            raw = self._content[(path, ref)]
        except KeyError as exc:
            raise EvalDriverError(
                f"fixture has no content for path={path!r} at ref={ref!r}; "
                f"check the file's status/content_base/content_head."
            ) from exc
        wrapped_b64 = base64.encodebytes(raw.encode()).decode("ascii")
        return _FixtureResponse(
            parsed_data=_FixtureContentFile(encoding="base64", content=wrapped_b64)
        )


class _FixturePullsAPI:
    def __init__(self, metas: list[_FixtureFileMeta]) -> None:
        self._metas = metas

    async def async_list_files(
        self, owner: str, repo: str, pull_number: int, **kwargs: Any
    ) -> _FixtureResponse:
        return _FixtureResponse(parsed_data=list(self._metas))


class _FixtureRestAPI:
    def __init__(self, repos: _FixtureReposAPI, pulls: _FixturePullsAPI) -> None:
        self.repos = repos
        self.pulls = pulls


class _FixtureGitHubClient:
    def __init__(self, rest: _FixtureRestAPI) -> None:
        self.rest = rest


# ---------------------------------------------------------------------------
# Adapter 3 — capturing publisher (GitHubPublisher Protocol; no POST)
# ---------------------------------------------------------------------------


@dataclass
class _CapturingPublisher:
    """Records `create_review` instead of POSTing; reports no prior review."""

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
            {"review_status": review_status, "comments": tuple(comments)}
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


# ---------------------------------------------------------------------------
# Adapter 4 — no-op import-path resolver (trace does not run in V1 scenarios)
# ---------------------------------------------------------------------------


class _NoOpImportPathResolver:
    """Satisfies build_graph's `ImportPathResolver` Protocol gate.

    Trace does not run in the V1 driven scenarios (their analyze responses
    carry no trace_candidates), so this is never actually called. A
    trace-exercising scenario (deferred) needs a real resolver.
    """

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        return []


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------


def _github_factory_for(fixture: EvalFixture) -> Any:
    """Build a `github_factory(installation_id)` serving the fixture's PR."""
    metas = [
        _FixtureFileMeta(
            filename=f.path,
            status=f.status,
            additions=f.additions,
            deletions=f.deletions,
            patch=f.patch,
            previous_filename=f.previous_path,
        )
        for f in fixture.files
    ]
    content: dict[tuple[str, str], str] = {}
    for f in fixture.files:
        # `previous_path` is the base-side path for renames; base content is read
        # there, head content at the new path (per intake's rename handling).
        base_path = f.previous_path if (f.status == "renamed" and f.previous_path) else f.path
        if f.content_base is not None:
            content[(base_path, fixture.base_sha)] = f.content_base
        if f.content_head is not None:
            content[(f.path, fixture.head_sha)] = f.content_head
    client = _FixtureGitHubClient(
        _FixtureRestAPI(_FixtureReposAPI(content), _FixturePullsAPI(metas))
    )

    def factory(installation_id: int) -> Any:
        if installation_id != fixture.installation_id:
            raise EvalDriverError(
                f"unexpected installation_id {installation_id} "
                f"(fixture is {fixture.installation_id})"
            )
        return client

    return factory


def _seed_state(fixture: EvalFixture, review_id: UUID) -> ReviewState:
    """Seed state with EMPTY changed_files — intake enriches via the fixture
    github client, exercising the real two-phase fetch (per DECISIONS.md#020)."""
    return ReviewState(
        review_id=review_id,
        received_at=datetime.now(UTC),
        pr_context=PRContext(
            installation_id=fixture.installation_id,
            owner=fixture.owner,
            repo=fixture.repo,
            pr_number=fixture.pr_number,
            base_sha=fixture.base_sha,
            head_sha=fixture.head_sha,
            pr_title=fixture.pr_title,
            pr_body=fixture.pr_body,
            author=fixture.author,
            total_additions=fixture.total_additions,
            total_deletions=fixture.total_deletions,
            changed_files=(),
        ),
        is_eval=True,
    )


async def _seed_installation(engine: AsyncEngine, fixture: EvalFixture) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations (installation_id, app_slug, account_id, "
                "account_login, account_type, permissions_at_install) "
                "VALUES (:id, 'eval-app', 1, :login, 'User', '{}'::jsonb) "
                "ON CONFLICT (installation_id) DO NOTHING"
            ),
            {"id": fixture.installation_id, "login": fixture.owner},
        )


async def _seed_review(engine: AsyncEngine, review_id: UUID, fixture: EvalFixture) -> None:
    """Seed the review row with `is_eval=True` (the integrity gate enforces it)."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO reviews (id, installation_id, repo_id, pr_number, head_sha, "
                "status, is_eval, files_examined, files_traced_beyond_diff, llm_calls_made, "
                "total_input_tokens, total_output_tokens, total_cost_usd, wall_clock_seconds, "
                "retention_expires_at) VALUES (:id, :iid, :repo, :pr, :sha, 'running', TRUE, "
                "0, 0, 0, 0, 0, 0, 0, NOW() + INTERVAL '180 days')"
            ),
            {
                "id": review_id,
                "iid": fixture.installation_id,
                "repo": _EVAL_REPO_ID,
                "pr": fixture.pr_number,
                "sha": fixture.head_sha,
            },
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _run_is_eval_gate(engine: AsyncEngine) -> None:
    """Run the is_eval integrity gate against a live connection on `engine`."""
    async with engine.connect() as conn:
        await assert_no_is_eval_violations(conn)


async def _drive(fixture: EvalFixture, db_url: str) -> EvalRunResult:
    """Run the graph once against `db_url` (already migrated) and collect results.

    The is_eval integrity gate runs on BOTH the success and failure paths before
    the ephemeral DB is dropped: on success a violation fails the run; on failure
    it is best-effort (suppressed) so it never masks the root-cause exception.
    """
    engine = create_async_engine(db_url, poolclass=NullPool)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        await _seed_installation(engine, fixture)
        review_id = uuid4()
        await _seed_review(engine, review_id, fixture)

        persister = AuditPersister(
            session_factory=session_factory, retention_settings=RetentionSettings()
        )
        publisher = _CapturingPublisher()
        provider = _FixtureScriptedProvider(fixture.llm_responses)
        graph = build_graph(
            db_factory=session_factory,
            github_factory=_github_factory_for(fixture),
            provider=provider,
            model_config=ModelConfig(),
            phase_event_sink=persister,
            file_examination_sink=persister,
            analyze_event_sink=persister,
            publish_event_sink=persister,
            trace_sink=persister,
            hitl_event_sink=persister,
            synthesize_event_sink=persister,
            review_status_sink=ReviewStatusPersister(session_factory=session_factory),
            anomaly_sink=AnomalyPersister(session_factory=session_factory),
            hitl_config=HITLConfig(),
            checkpointer=InMemorySaver(),
            publisher=publisher,
            import_path_resolver=_NoOpImportPathResolver(),
        )

        result = await graph.ainvoke(
            _seed_state(fixture, review_id),
            config={"configurable": {"thread_id": str(review_id)}},
        )

        # A CRITICAL/HIGH finding trips the HITL gate -> the graph interrupt()s
        # and LangGraph returns the accumulated channel state PLUS `__interrupt__`
        # (a list of Interrupt objects; `.value` is hitl's request payload). We
        # record the gate but do NOT resume — single-pass driver (see EvalRunResult).
        interrupts = result.get("__interrupt__")
        hitl_gated = bool(interrupts)
        hitl_request = HITLRequest.model_validate(interrupts[0].value) if interrupts else None

        report = result.get("review_report")
        findings: tuple[ReviewFinding, ...] = tuple(report.findings) if report else ()
        trace_decisions: tuple[TraceDecision, ...] = tuple(result.get("trace_decisions", ()))
        published: tuple[InlineComment, ...] = tuple(
            c for call in publisher.create_review_calls for c in call["comments"]
        )
        eval_result = EvalRunResult(
            review_id=review_id,
            findings=findings,
            trace_decisions=trace_decisions,
            published_comments=published,
            hitl_gated=hitl_gated,
            hitl_request=hitl_request,
        )

        # is_eval integrity gate (success path): a violation fails the run.
        await _run_is_eval_gate(engine)
        return eval_result
    except Exception:
        # Failed run: still verify is_eval discipline before the DB is dropped,
        # but never let a gate violation mask the root-cause exception.
        with contextlib.suppress(Exception):
            await _run_is_eval_gate(engine)
        raise
    finally:
        await engine.dispose()


async def _arun_review(fixture: EvalFixture, base_url: str) -> EvalRunResult:
    async with ephemeral_database(base_url=base_url) as db_url:
        await run_alembic_upgrade_head(db_url)
        return await _drive(fixture, db_url)


def run_review(fixture_path: str | os.PathLike[str]) -> EvalRunResult:
    """Drive the real 7-node graph against a JSON PR fixture; return the result.

    Synchronous wrapper (the eval scenarios call it without `await`); the async
    graph runs inside `asyncio.run`. Fail-closed: requires `OUTRIDER_IS_EVAL=1`
    and a port-5433/"test" `TEST_DATABASE_URL` before any database work.
    """
    require_eval_mode()
    try:
        base_url = os.environ[_TEST_DB_URL_ENV_VAR]
    except KeyError as exc:
        raise EvalDriverError(
            f"{_TEST_DB_URL_ENV_VAR} is not set; the eval driver needs the "
            "postgres-test URL. Run under pytest with --is-eval after "
            "`source .env`, or export it explicitly."
        ) from exc

    with open(fixture_path, encoding="utf-8") as fh:
        fixture = EvalFixture.model_validate(json.load(fh))

    return asyncio.run(_arun_review(fixture, base_url))
