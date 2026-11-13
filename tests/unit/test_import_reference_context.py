"""Bounded acquisition uses pinned GitHub source without changing PR inventory."""

import base64
import difflib
import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from outrider.agent.nodes.analyze_context import PreparedReviewContext, prepare_review_context
from outrider.agent.nodes.import_reference_context import prepare_import_reference_contexts
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_state import ReviewState

_HEAD_SHA = "b" * 40


class _GitHubHTTPError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"GitHub HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class _RepositoryContents:
    """The wrapper's real inline-base64 shape, including GitHub's line wrapping."""

    def __init__(self, contents: dict[str, str | bytes | None | Exception]) -> None:
        self.contents = contents
        self.requests: list[tuple[str, str, str, str]] = []
        self.client = SimpleNamespace(
            rest=SimpleNamespace(repos=SimpleNamespace(async_get_content=self.get_content))
        )
        self.factory = AsyncMock(return_value=self.client)

    async def get_content(self, owner: str, repo: str, path: str, *, ref: str) -> SimpleNamespace:
        self.requests.append((owner, repo, path, ref))
        if path not in self.contents:
            raise _GitHubHTTPError(404)
        content = self.contents[path]
        if isinstance(content, Exception):
            raise content
        if content is None:
            data = SimpleNamespace(encoding="none", content=None)
        else:
            raw = content.encode() if isinstance(content, str) else content
            data = SimpleNamespace(encoding="base64", content=base64.encodebytes(raw).decode())
        return SimpleNamespace(parsed_data=data)


def _changed(path: str, head: str) -> ChangedFile:
    before = head.replace("return ", "return None # ", 1)
    patch = "".join(
        list(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                head.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )[2:]
    )
    return ChangedFile(
        path=path,
        status="modified",
        additions=1,
        deletions=1,
        patch=patch,
        content_base=before,
        content_head=head,
    )


def _review(*files: ChangedFile) -> ReviewState:
    return ReviewState(
        review_id=UUID("00000000-0000-0000-0000-000000000001"),
        received_at=datetime(2026, 11, 3, tzinfo=UTC),
        pr_context=PRContext(
            installation_id=77,
            owner="owner",
            repo="project",
            pr_number=4,
            pr_title="Change decoding",
            base_sha="a" * 40,
            head_sha=_HEAD_SHA,
            author="author",
            changed_files=files,
            total_additions=sum(file.additions for file in files),
            total_deletions=sum(file.deletions for file in files),
        ),
    )


def _base_contexts(
    state: ReviewState, files: tuple[ChangedFile, ...]
) -> dict[str, PreparedReviewContext]:
    return {
        file.path: prepare_review_context(
            file_path=file.path,
            content=file.content_head,
            patch=file.patch,
            changed_files=state.pr_context.changed_files,
            import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
        )
        for file in files
    }


async def _prepare(state: ReviewState, files: tuple[ChangedFile, ...], github: _RepositoryContents):
    return await prepare_import_reference_contexts(
        state,
        files=files,
        base_contexts=_base_contexts(state, files),
        github_factory=github.factory,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )


async def test_alias_reference_is_fetched_at_pinned_head_with_exact_provenance_and_no_mutation():
    primary = _changed(
        "pkg/handler.py",
        "from .decoder import decode as convert\ndef changed(value):\n    return convert(value)\n",
    )
    source = (
        "@decorate\n"
        "def decode(value):\n"
        '    """Keep this documentation and the complete guard."""\n'
        "    if value is None:\n"
        "        raise ValueError('invalid')\n"
        "    return value\n"
        "\n"
        "def unrelated():\n"
        "    return 'not requested'\n"
    )
    state = _review(primary)
    before = state.model_dump_json()
    base = _base_contexts(state, (primary,))
    github = _RepositoryContents({"pkg/decoder.py": source})

    prepared = await prepare_import_reference_contexts(
        state,
        files=(primary,),
        base_contexts=base,
        github_factory=github.factory,
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )

    github.factory.assert_awaited_once_with(77)
    assert github.requests == [
        ("owner", "project", "pkg/decoder.py", _HEAD_SHA),
        ("owner", "project", "pkg/decoder/__init__.py", _HEAD_SHA),
    ]
    context = prepared[primary.path]
    assert _HEAD_SHA in context.text
    assert hashlib.sha256(source.encode()).hexdigest() in context.text
    assert '"path": "pkg/decoder.py"' in context.text
    assert "1 | @decorate\n2 | def decode(value):" in context.text
    assert "5 |         raise ValueError('invalid')\n6 |     return value" in context.text
    assert "def unrelated" not in context.text
    reference = [item for item in context.manifest if item.inclusion_reason == "captured_reference"]
    assert [
        (item.file_path, item.source_side, item.line_start, item.line_end) for item in reference
    ] == [("pkg/decoder.py", "head", 1, 6)]
    assert state.model_dump_json() == before
    assert _base_contexts(state, (primary,)) == base


@pytest.mark.parametrize("counterpart", ["present", "unusable", "absent"])
async def test_unique_reference_requires_a_known_absent_counterpart(counterpart: str):
    primary = _changed(
        "pkg/handler.py",
        "from .decoder import decode\ndef changed(value):\n    return decode(value)\n",
    )
    contents = {"pkg/decoder.py": "def decode(value):\n    return value\n"}
    if counterpart == "present":
        contents["pkg/decoder/__init__.py"] = "def decode(value):\n    return other(value)\n"
    elif counterpart == "unusable":
        contents["pkg/decoder/__init__.py"] = None
    github = _RepositoryContents(contents)
    context = (await _prepare(_review(primary), (primary,), github))[primary.path]
    references = [
        item for item in context.manifest if item.inclusion_reason == "captured_reference"
    ]

    assert len(github.requests) == 2
    assert bool(references) is (counterpart == "absent")
    if counterpart != "absent":
        assert "ambiguous or unavailable module" in context.text
        assert "def decode(value):" not in context.text


async def test_reference_reads_are_shared_across_workers_and_stop_after_the_second_hop():
    head = "from .decoder import decode\ndef changed(value):\n    return decode(value)\n"
    files = (_changed("pkg/first.py", head), _changed("pkg/second.py", head))
    state = _review(*files)
    github = _RepositoryContents(
        {
            "pkg/decoder.py": (
                "from .policy import check\ndef decode(value):\n    return check(value)\n"
            ),
            "pkg/policy.py": (
                "from .third import final\ndef check(value):\n    return final(value)\n"
            ),
            "pkg/third.py": "def final(value):\n    return value\n",
        }
    )
    prepared = await _prepare(state, files, github)

    assert [request[2] for request in github.requests] == [
        "pkg/decoder.py",
        "pkg/decoder/__init__.py",
        "pkg/policy.py",
        "pkg/policy/__init__.py",
    ]
    github.factory.assert_awaited_once_with(77)
    for context in prepared.values():
        references = [
            entry.file_path
            for entry in context.manifest
            if entry.inclusion_reason == "captured_reference"
        ]
        assert references == ["pkg/decoder.py", "pkg/policy.py"]
        assert "dependency depth limit" in context.text
        assert "def final(value):" not in context.text


async def test_attempted_reads_and_retained_reference_files_stay_bounded():
    names = [f"helper{number}" for number in range(9)]
    imports = "".join(f"from .{name} import {name}\n" for name in names)
    head = (
        imports
        + "def changed(value):\n    return "
        + " + ".join(f"{name}(value)" for name in names)
        + "\n"
    )
    primary = _changed("pkg/handler.py", head)
    github = _RepositoryContents(
        {f"pkg/{name}.py": f"def {name}(value):\n    return value\n" for name in names}
    )
    context = (await _prepare(_review(primary), (primary,), github))[primary.path]
    references = [
        item for item in context.manifest if item.inclusion_reason == "captured_reference"
    ]

    assert len(github.requests) == 16
    assert len(references) == 4
    assert {item.file_path for item in references} == {
        f"pkg/helper{number}.py" for number in range(4)
    }
    assert "retained source limit" in context.text
    assert "ambiguous or unavailable module" in context.text
    assert all(request[2] != "pkg/helper8.py" for request in github.requests)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [(b"\xff", "non-UTF8 source"), (b"#" * (128 * 1024 + 1), "source size limit")],
)
async def test_unusable_reference_source_is_omitted_without_becoming_a_primary_file(raw, reason):
    primary = _changed(
        "pkg/handler.py",
        "from .decoder import decode\ndef changed(value):\n    return decode(value)\n",
    )
    state = _review(primary)
    before = state.model_dump_json()
    github = _RepositoryContents({"pkg/decoder.py": raw})
    context = (await _prepare(state, (primary,), github))[primary.path]

    assert not any(item.inclusion_reason == "captured_reference" for item in context.manifest)
    assert reason in context.text
    assert state.model_dump_json() == before


@pytest.mark.parametrize("status", [403, 500])
async def test_non404_github_failures_propagate_instead_of_becoming_missing_source(status):
    primary = _changed(
        "pkg/handler.py",
        "from .decoder import decode\ndef changed(value):\n    return decode(value)\n",
    )
    github = _RepositoryContents({"pkg/decoder.py": _GitHubHTTPError(status)})
    with pytest.raises(_GitHubHTTPError, match=str(status)):
        await _prepare(_review(primary), (primary,), github)
    assert [request[2] for request in github.requests] == ["pkg/decoder.py"]


async def test_non_python_files_make_no_reference_requests():
    primary = _changed("web/handler.ts", "export function changed(value) { return value; }\n")
    github = _RepositoryContents({})
    prepared = await _prepare(_review(primary), (primary,), github)
    assert prepared == {}
    assert github.requests == []
    github.factory.assert_not_awaited()


async def test_already_supplied_head_source_is_reused_without_refetching_the_changed_file():
    primary = _changed(
        "pkg/handler.py",
        "from .decoder import decode\ndef changed(value):\n    return decode(value)\n",
    )
    decoder = ChangedFile(
        path="pkg/decoder.py",
        status="added",
        additions=2,
        deletions=0,
        content_head="def decode(value):\n    return value\n",
    )
    state = _review(primary, decoder)
    before = state.model_dump_json()
    github = _RepositoryContents({})
    context = (await _prepare(state, (primary,), github))[primary.path]

    assert [request[2] for request in github.requests] == ["pkg/decoder/__init__.py"]
    assert any(item.file_path == decoder.path for item in context.manifest)
    assert state.model_dump_json() == before
