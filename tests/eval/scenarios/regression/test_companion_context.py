"""Check sibling-source delivery and finding admission through the real graph.

Responses are scripted: this proves orchestration, not model detection quality.
Requires --is-eval and postgres-test; no external service is contacted.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from outrider.agent import run_review
from outrider.agent.eval_adapters import _FixtureReposAPI
from outrider.agent.eval_driver import _FixtureScriptedProvider

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from outrider.agent.eval_adapters import _FixtureResponse
    from outrider.llm.base import LLMExchangePersister, LLMRequest, LLMResponse


def test_companion_source_reaches_worker_without_expanding_finding_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementation = "def toggle(disabled: bool) -> bool:\n    return bool(disabled)\n"
    declaration = "def toggle(run: bool) -> bool: ...\n"
    finding = {
        "finding_type": "correctness_bug",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "Declared keyword is rejected at runtime",
        "description": "The declaration accepts toggle(run=True), but the implementation's "
        "parameter is disabled, so that call raises TypeError.",
        "evidence": declaration.rstrip(),
        "line_start": 1,
        "line_end": 1,
        "trace_candidates": [],
    }
    responses = {
        "triage": [
            json.dumps(
                {
                    "file_tiers": {"pkg.py": "skim", "pkg.pyi": "deep"},
                    "overall_risk": "medium",
                    "relevant_dimensions": ["code_quality"],
                    "reasoning": "Review the public declaration.",
                }
            )
        ],
        "analyze": [
            json.dumps(
                {
                    "findings": [
                        finding,
                        {
                            **finding,
                            "title": "Reference line must not become a primary location",
                            "evidence": "    return bool(disabled)",
                            "line_start": 2,
                            "line_end": 2,
                        },
                    ]
                }
            )
        ],
        "synthesize": ["One correctness bug in the public keyword declaration."],
    }
    fixture = {
        "installation_id": 7401,
        "owner": "acme",
        "repo": "switches",
        "pr_number": 1,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "pr_title": "Add toggle declaration",
        "author": "contributor",
        "total_additions": 3,
        "total_deletions": 0,
        "files": [
            {
                "path": path,
                "status": "added",
                "additions": len(source.splitlines()),
                "deletions": 0,
                "content_head": source,
                "patch": f"@@ -0,0 +1,{len(source.splitlines())} @@\n"
                + "".join("+" + line for line in source.splitlines(keepends=True)),
            }
            for path, source in (("pkg.py", implementation), ("pkg.pyi", declaration))
        ],
        "llm_responses": {},
    }
    fixture_path = tmp_path / "companion.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    requests: list[LLMRequest] = []

    class RecordingProvider(_FixtureScriptedProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            requests.append(request)
            return await super().complete(request)

    def provider_factory(persister: LLMExchangePersister) -> RecordingProvider:
        return RecordingProvider(
            responses,
            persister=persister,
            triage_paths=tuple(file["path"] for file in fixture["files"]),
        )

    fetches: list[tuple[str, str]] = []
    original_fetch = _FixtureReposAPI.async_get_content

    async def record_fetch(
        client: _FixtureReposAPI, owner: str, repo: str, path: str, *, ref: str
    ) -> _FixtureResponse:
        fetches.append((path, ref))
        return await original_fetch(client, owner, repo, path, ref=ref)

    monkeypatch.setattr(_FixtureReposAPI, "async_get_content", record_fetch)
    result = run_review(
        fixture_path,
        companion_context_enabled=True,
        provider_factory=provider_factory,
    )

    (analysis,) = [request for request in requests if request.node_id == "analyze"]
    assert "def toggle(disabled: bool)" in analysis.user_prompt
    assert any(
        entry.file_path == "pkg.py" and entry.inclusion_reason == "captured_reference"
        for entry in analysis.context_summary
    )
    assert {
        (entry.file_path, entry.line_start, entry.line_end)
        for entry in analysis.context_summary
        if entry.inclusion_reason in {"changed_scope", "changed_region"}
    } == {("pkg.pyi", 1, 1)}
    assert sorted(fetches) == [("pkg.py", "b" * 40), ("pkg.pyi", "b" * 40)]
    (admitted,) = result.findings
    assert admitted.title == finding["title"]
    assert (admitted.file_path, admitted.line_start, admitted.line_end) == ("pkg.pyi", 1, 1)
    assert admitted.evidence_tier == "judged"
    assert result.published_comments
