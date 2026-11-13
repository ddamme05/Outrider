"""Native checker preparation must retain complete, eligible implementation evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from outrider.agent.candidate_checks import prepare_candidate_check
from outrider.agent.native_candidates import collect_native_candidates
from outrider.agent.nodes.analyze_parser import parse_analyze_response
from outrider.ast_facts.registry import parse_source
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER
from outrider.llm.base import LLMResponse
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_state import ReviewState

STUB_PATH = "pkg/codec.pyi"
STUB = "def encode(value: bytes) -> bytes: ...\n"
RUNTIME = "def encode(payload):\n    return canonicalize(payload)\n"
HELPER = "def canonicalize(value):\n    return b'' if value is None else value\n"
MODEL = "claude-sonnet-5-5"


def prepare(
    sources: dict[str, str],
    *,
    primary_path: str = STUB_PATH,
    primary_source: str = STUB,
    proposal: dict | None = None,
):
    now = datetime(2026, 11, 7, tzinfo=UTC)
    review_id = UUID("1baa27d9-4aa2-4f8e-b53f-11328d621a8a")
    sources = {primary_path: primary_source, **sources}
    proposal = proposal or {
        "finding_type": "correctness_bug",
        "evidence_tier": "judged",
        "title": "Declared keyword differs from the implementation",
        "description": (
            "The declaration accepts value but encode uses payload. "
            "Rename the declaration's parameter to payload."
        ),
        "evidence": STUB.rstrip("\n"),
        "line_start": 1,
        "line_end": 1,
    }
    response = LLMResponse(
        text=json.dumps({"findings": [proposal]}),
        model=MODEL,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        latency_ms=0,
        finish_reason="end_turn",
        exchange_event_id=UUID("aac15d04-ec6e-43b6-8ab7-e290b55983d3"),
        exchange_recorded_at=now,
        profile_id="anthropic",
        reasoning_enabled=False,
        profile_contract_digest="a" * 64,
    )
    facts = parse_source(primary_source.encode(), primary_path, COORDINATES_IMPORT_PATH_RESOLVER)
    parsed = parse_analyze_response(
        response.text,
        review_id=review_id,
        installation_id=7,
        file_path=primary_path,
        file_content=primary_source,
        file_byte_length=len(primary_source.encode()),
        included_scope_units=facts.scope_units,
        query_match_id_set=frozenset(),
        degraded_mode=False,
        active_policy_version=ACTIVE_POLICY_VERSION,
        primary_line_ranges=((1, 1),),
    )
    assert len(parsed.admitted_findings) == 1
    (ref,) = collect_native_candidates(
        response, parsed, review_id=review_id, file_path=primary_path, pass_index=0
    )
    state = ReviewState(
        review_id=review_id,
        received_at=now,
        is_eval=True,
        pr_context=PRContext(
            installation_id=7,
            owner="owner",
            repo="repo",
            pr_number=1,
            pr_title="Update a declaration",
            head_sha="a" * 40,
            base_sha="b" * 40,
            author="developer",
            total_additions=sum(len(source.splitlines()) for source in sources.values()),
            total_deletions=0,
            changed_files=tuple(
                ChangedFile(
                    path=path,
                    status="added",
                    additions=len(source.splitlines()),
                    deletions=0,
                    content_head=source,
                    language="python",
                )
                for path, source in sources.items()
            ),
        ),
        native_candidates=[ref],
    )
    original = ref.model_dump(mode="json")
    prepared = prepare_candidate_check(
        state,
        ref,
        response.text,
        model=MODEL,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    assert prepared is not None
    assert ref.model_dump(mode="json") == original
    user = json.loads(prepared.request.user_prompt)
    rendered = json.dumps(
        {key: user[key] for key in ("reference_snippets", "change_context", "limits")},
        ensure_ascii=False,
    )
    assert len(rendered.encode()) <= 12_000
    assert len(prepared.check.snippets) <= 8
    for snippet in prepared.check.snippets:
        source = sources[snippet.file_path]
        text = "\n".join(source.splitlines()[snippet.line_start - 1 : snippet.line_end])
        assert len(text.encode()) <= 2_000
        assert snippet.source_hash == hashlib.sha256(source.encode()).hexdigest()
        assert snippet.text_hash == hashlib.sha256(text.encode()).hexdigest()
    return prepared, user


def test_called_default_implementations_reach_checker_as_possible_references():
    source = (
        "def archive_record(owner, item):\n"
        "    owner.archived.append(item)\n"
        "\n"
        "class RemoteWriter:\n"
        "    def send(self, item):\n"
        "        self.sent.append(item)\n"
        "\n"
        "class BufferedWriter:\n"
        "    def add(self, item, archive=archive_record, dispatch=RemoteWriter.send):\n"
        "        archive(self, item)\n"
        "        dispatch(self, item)\n"
        "        self.pending.append(item)\n"
    )
    _, user = prepare(
        {},
        primary_path="pkg/writer.py",
        primary_source=source,
        proposal={
            "finding_type": "correctness_bug",
            "evidence_tier": "judged",
            "title": "Appending queues the same record three times",
            "description": "Both callbacks write into pending before add appends it again.",
            "evidence": "        dispatch(self, item)",
            "line_start": 11,
            "line_end": 11,
        },
    )
    snippets = user["reference_snippets"]
    assert (snippets[0]["line_start"], snippets[0]["line_end"]) == (9, 12)
    by_range = {(item["line_start"], item["line_end"]): item for item in snippets[1:]}
    assert by_range.keys() >= {(1, 2), (5, 6)}
    assert "owner.archived.append(item)" in by_range[1, 2]["source"]
    assert "self.sent.append(item)" in by_range[5, 6]["source"]
    assert all(
        "possible default implementation" in by_range[span]["selection_reason"]
        for span in ((1, 2), (5, 6))
    )


def test_small_companion_reaches_native_request_without_expanding_dependencies():
    prepared, user = prepare({"pkg/codec.py": RUNTIME, "pkg/helpers.py": HELPER})
    snippets = {snippet.file_path: snippet for snippet in prepared.check.snippets}
    assert set(snippets) == {STUB_PATH, "pkg/codec.py"}
    assert (snippets["pkg/codec.py"].line_start, snippets["pkg/codec.py"].line_end) == (1, 2)
    rendered = {snippet["file_path"]: snippet["source"] for snippet in user["reference_snippets"]}
    assert (
        rendered["pkg/codec.py"] == "1 | def encode(payload):\n2 |     return canonicalize(payload)"
    )


@pytest.mark.parametrize(
    "sources",
    [
        {"elsewhere/codec.py": RUNTIME},
        {"pkg/codec.py": RUNTIME + "\ndef encode(other):\n    return other\n"},
        {
            "pkg/codec.py": (
                "def encode(payload):\n    note = "
                + repr("é" * 1_100)
                + "\n    return canonicalize(payload)\n"
            )
        },
    ],
    ids=["missing-exact-sibling", "ambiguous-definition", "over-checker-byte-cap"],
)
def test_ineligible_companion_is_omitted_whole_with_primary_preserved(sources):
    prepared, user = prepare({**sources, "pkg/helpers.py": HELPER})
    assert [
        (snippet.file_path, snippet.line_start, snippet.line_end)
        for snippet in prepared.check.snippets
    ] == [(STUB_PATH, 1, 1)]
    assert user["reference_snippets"][0]["source"] == "1 | def encode(value: bytes) -> bytes: ..."


@pytest.mark.parametrize("caller_padding", [0, 1_341], ids=["snippet-cap", "rendered-byte-cap"])
def test_saturated_packet_preserves_existing_evidence_instead_of_adding_companion(caller_padding):
    callers = {
        f"pkg/caller{number}.py": (
            f"def caller{number}():\n    return encode(b'')  # " + "x" * caller_padding + "\n"
        )
        for number in range(7)
    }
    prepared, user = prepare({"pkg/codec.py": RUNTIME, "pkg/helpers.py": HELPER, **callers})
    assert [snippet.file_path for snippet in prepared.check.snippets] == [STUB_PATH, *callers]
    assert len(user["reference_snippets"]) == 8
