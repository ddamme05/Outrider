"""Execute captured source to check labels without model-generated answers.

Upload tests execute the complete captured module with installed Starlette's
thread-pool helper and real local spooled files. OAuth tests execute the exact
captured callback and identity-resolver bodies with explicit framework facades; they
do not reproduce a complete Flask deployment or a live OAuth exchange.
"""

from __future__ import annotations

import ast
import json
import tempfile
import threading
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

import pytest

from outrider.agent.eval_driver import EvalFixture

from . import direct_review, real_pr_cases
from .review_cases import (
    DATASET_VERSION,
    EXPANDED_DATASET_VERSION,
    ROOT,
    dataset_inputs,
    load_cases,
)


def _source(case_id: str) -> str:
    case = next(case for case in load_cases() if case.id == case_id)
    fixture = EvalFixture.model_validate_json((ROOT / case.fixture_path).read_bytes())
    return next(file.content_head for file in fixture.files if file.path == case.analysis_file)


class _TrackingSpool(tempfile.SpooledTemporaryFile):
    def rollover(self) -> None:
        self.rollover_thread = threading.get_ident()
        super().rollover()


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", ["starlette-933", "starlette-2962"])
async def test_captured_upload_rollover_thread(case_id: str) -> None:
    namespace = {"__name__": "reviewed_captured_datastructures"}
    # Trusted, hash-checked public source, never a model-produced program.
    exec(compile(_source(case_id), case_id, "exec"), namespace)  # noqa: S102
    with _TrackingSpool(max_size=4) as spool:
        upload = namespace["UploadFile"](filename="sample", file=spool)
        await upload.write(b"0123456789")
        assert spool._rolled
        assert (spool.rollover_thread == threading.get_ident()) == (case_id == "starlette-933")
        # An ordinary already-spooled write must also succeed in both programs.
        await upload.write(b"a")
        spool.seek(0)
        assert spool.read() == b"0123456789a"


def _oauth_callback(case_id: str, resolved_email: str | None):
    source = _source(case_id)
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_oauth_response_common", "oauth_verify_response"}
    ]
    assert len(functions) == 2 and all(not node.decorator_list for node in functions)
    body = "\n\n".join(ast.get_source_segment(source, node) for node in functions)
    session = {"fs_paa": 1, "fs_oauth_next": "/profile"}
    user = (
        SimpleNamespace(email=resolved_email, get_redirect_qparams=lambda: {})
        if resolved_email is not None
        else None
    )
    provider = SimpleNamespace(
        fetch_identity_cb=lambda _app, _token: ("email", resolved_email or "unknown@example.com")
    )
    oauth = SimpleNamespace(
        providers={"github": provider},
        authlib_provider=lambda _name: SimpleNamespace(authorize_access_token=lambda: {}),
        oauth_app=object(),
    )
    namespace = {
        "_security": SimpleNamespace(
            oauthglue=oauth, datastore=SimpleNamespace(find_user=lambda **_kwargs: user)
        ),
        "get_identity_attribute": lambda _name: {"mapper": lambda value: value},
        "OAuthError": RuntimeError,
        "current_user": SimpleNamespace(email="victim@example.com"),
        "session": session,
        "time": SimpleNamespace(time=lambda: 1000),
        "cv": lambda name: {"REDIRECT_BEHAVIOR": "spa"}.get(name, name),
        "get_url": lambda view, **_kwargs: view,
        "get_message": lambda *_args, **_kwargs: ("message", "error"),
        "redirect": lambda target: (302, target),
    }
    exec(compile("from __future__ import annotations\n" + body, case_id, "exec"), namespace)  # noqa: S102
    response = namespace["oauth_verify_response"]("github")
    return session, response


@pytest.mark.parametrize("case_id", ["flask-security-1183", "flask-security-1216"])
def test_captured_oauth_freshness_requires_same_identity(case_id: str) -> None:
    same, response = _oauth_callback(case_id, "victim@example.com")
    assert same["fs_paa"] == 1000 and response[0] == 302
    unknown, _ = _oauth_callback(case_id, None)
    assert unknown["fs_paa"] == 1
    different, response = _oauth_callback(case_id, "attacker@example.com")
    assert different["fs_paa"] == (1000 if case_id == "flask-security-1183" else 1)
    assert "fs_oauth_next" not in different
    assert response[0] == 302


def test_review_cases_keep_provenance_and_mechanism_labels_separate() -> None:
    cases = load_cases()
    assert len(cases) == 4
    assert len({case.provenance_family for case in cases}) == 2
    assert sum(bool(case.expected) for case in cases) == 2
    for case in cases:
        fixture = EvalFixture.model_validate_json((ROOT / case.fixture_path).read_bytes())
        assert not fixture.llm_responses
        assert fixture.analyze_responses_by_path is None
        assert not fixture.repository_contents_head
        assert case.exposure == "development-exposed"
        assert bool(case.expected) != bool(case.scoped_counterparts)
    pins = dataset_inputs()
    assert all(len(digest) == 64 for digest in pins.values())
    assert "tests/eval/test_review_case_semantics.py" in pins
    manifest = json.loads((ROOT / "tests/eval/corpus/review_comparison/manifest.json").read_text())
    assert manifest["held_out"] is False
    assert "not captured" in manifest["beyond_diff_context"].lower()


def test_expanded_cases_preserve_original_captures_and_reviewed_labels() -> None:
    original = load_cases()
    expanded = load_cases(include_path_cases=True)
    assert expanded[: len(original)] == original
    assert len(expanded) == 7
    assert len({case.provenance_family for case in expanded}) == 3
    assert DATASET_VERSION != EXPANDED_DATASET_VERSION
    path_cases = real_pr_cases.load_cases()
    assert [case.id for case in expanded[len(original) :]] == [case.id for case in path_cases]
    for adapted, source in zip(expanded[len(original) :], path_cases, strict=True):
        assert adapted.fixture_path == source.fixture_path
        assert adapted.fixture_sha256 == source.fixture_sha256
        assert adapted.analysis_file == source.analysis_file
        assert adapted.expected == source.expected
        assert [asdict(scope) for scope in adapted.scoped_counterparts] == [
            asdict(scope) for scope in source.scoped_counterparts
        ]
        assert adapted.capture_manifest_path == source.receipt_path
        assert adapted.provenance_family == source.provenance_family
        assert adapted.provenance_source == source.provenance_source
        assert adapted.exposure == source.exposure
        assert (adapted.context_status, adapted.context_reason) == (
            source.context_status,
            source.context_reason,
        )
        fixture = EvalFixture.model_validate_json((ROOT / adapted.fixture_path).read_bytes())
        receipt = json.loads((ROOT / adapted.capture_manifest_path).read_text())
        assert {file.path for file in fixture.files} == {file["path"] for file in receipt["files"]}
        assert not fixture.llm_responses
        assert fixture.analyze_responses_by_path is None
        request = direct_review.render_request(fixture, model="unused", review_id=uuid4())
        metadata = json.loads(request.user_prompt.split("\n\n", 2)[1])
        assert metadata == {"pr_title": fixture.pr_title}
        assert adapted.context_reason not in request.user_prompt
        assert adapted.exposure not in request.user_prompt
        assert adapted.provenance_source not in request.user_prompt
        for file in fixture.files:
            assert f"File: {json.dumps(file.path)}" in request.user_prompt
            if file.patch:
                assert file.patch.rstrip("\n") in request.user_prompt


def test_expanded_dataset_pins_both_original_input_inventories() -> None:
    original = dataset_inputs()
    assert not any("real_pr_refresh/" in path for path in original)
    expanded = dataset_inputs(include_path_cases=True)
    assert original.items() <= expanded.items()
    assert real_pr_cases.dataset_inputs().items() <= expanded.items()
    for case in load_cases(include_path_cases=True):
        assert expanded[case.fixture_path] == case.fixture_sha256
        assert case.capture_manifest_path in expanded
