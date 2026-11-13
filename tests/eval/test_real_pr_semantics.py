"""Check captured upstream behavior with local files, without model calls.

Starlette response handling is reduced to a response carrier. The dbt legacy
branch uses an explicit capability facade rather than an older interpreter.
"""

from __future__ import annotations

import ast
import io
import json
import os
import stat
import sys
import tarfile
import typing
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.responses import PlainTextResponse, Response

_CORPUS = Path(__file__).with_name("corpus") / "real_pr_refresh"
_HEADS = {
    "starlette-985": "a260a36c3c75225f7bb3debe21de6744d3b46cc6",
    "dbt-common-182": "5656f48554842d13305c395dc7f0147f73ca9ac4",
    "dbt-common-353": "882ac14d320f8779fcb461c059371f7f7e63d848",
}


def _source(case: str, path: str, side: str = "head") -> str:
    fixture = json.loads((_CORPUS / case / "fixture.json").read_text())
    assert fixture["head_sha"] == _HEADS[case]
    return next(file[f"content_{side}"] for file in fixture["files"] if file["path"] == path)


async def _aio_stat(path: str) -> os.stat_result:
    return os.stat(path)


def _static_app(source: str, directory: Path):
    tree = ast.parse(source)
    original = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StaticFiles"
    )
    methods = [
        node
        for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"get_path", "get_response", "lookup_path"}
    ]
    selected = ast.ClassDef(
        name="CapturedStaticFiles", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    namespace = {
        "os": os,
        "stat": stat,
        "typing": typing,
        "Scope": dict,
        "Response": Response,
        "PlainTextResponse": PlainTextResponse,
        "aio_stat": _aio_stat,
    }
    exec(  # noqa: S102 -- Run selected captured methods without importing the project.
        compile(
            ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])),
            "<captured StaticFiles>",
            "exec",
        ),
        namespace,
    )
    app = namespace["CapturedStaticFiles"]()
    app.all_directories = [str(directory)]
    app.html = False
    app.file_response = lambda path, result, scope: PlainTextResponse(Path(path).read_text())
    return app


def _safe_extract(source: str, api):
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "safe_extract"
    )
    namespace = {"os": os, "tarfile": api}
    exec(  # noqa: S102 -- Run the captured function without importing the project.
        compile(ast.Module(body=[node], type_ignores=[]), "<captured safe_extract>", "exec"),
        namespace,
    )
    return namespace["safe_extract"]


@contextmanager
def _tarball(member_name: str, *, legacy_default: bool = False, symlink: bool = False):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as writer:
        if symlink:
            link = tarfile.TarInfo("redirect")
            link.type = tarfile.SYMTYPE
            link.linkname = "../destevil"
            writer.addfile(link)
        member = tarfile.TarInfo(member_name)
        data = b"controlled marker"
        member.size = len(data)
        writer.addfile(member, io.BytesIO(data))
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r:") as archive:
        if legacy_default:
            archive.extraction_filter = tarfile.fully_trusted_filter
        yield archive


@pytest.mark.asyncio
async def test_starlette_forward_pr_exposes_a_same_prefix_sibling(tmp_path: Path) -> None:
    directory = tmp_path / "static"
    directory.mkdir()
    (directory / "ok.txt").write_text("inside")
    (tmp_path / "static1.txt").write_text("outside marker")
    responses = []
    for side in ("base", "head"):
        app = _static_app(_source("starlette-985", "starlette/staticfiles.py", side), directory)
        scope = {"method": "GET", "path": "/../static1.txt"}
        responses.append(await app.get_response(app.get_path(scope), scope))
        scope["path"] = "/ok.txt"
        control = await app.get_response(app.get_path(scope), scope)
        assert control.status_code == 200 and control.body == b"inside"
    assert responses[0].status_code == 404
    assert responses[1].status_code == 200 and responses[1].body == b"outside marker"


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 13),
    reason="This oracle pins Python 3.13's unfiltered extraction default.",
)
def test_dbt_incomplete_fix_writes_to_a_same_prefix_sibling(tmp_path: Path) -> None:
    extract = _safe_extract(_source("dbt-common-182", "dbt_common/clients/system.py"), tarfile)
    destination = tmp_path / "dest"
    destination.mkdir()
    assert hasattr(tarfile, "data_filter") and not hasattr(tarfile.TarFile, "data_filter")
    with _tarball("../destevil/payload.txt") as archive:
        extract(archive, str(destination))
    assert (tmp_path / "destevil/payload.txt").read_bytes() == b"controlled marker"
    with _tarball("ok.txt") as archive:
        extract(archive, str(destination))
    assert (destination / "ok.txt").read_bytes() == b"controlled marker"


def test_dbt_fix_blocks_prefix_escape_without_proving_the_legacy_branch_safe(
    tmp_path: Path,
) -> None:
    source = _source("dbt-common-353", "dbt_common/clients/system.py")
    legacy = SimpleNamespace(
        TarFile=tarfile.TarFile, OutsideDestinationError=tarfile.OutsideDestinationError
    )
    for name, api in (("modern", tarfile), ("legacy", legacy)):
        destination = tmp_path / name / "dest"
        destination.mkdir(parents=True)
        extract = _safe_extract(source, api)
        with (
            _tarball("../destevil/payload.txt", legacy_default=name == "legacy") as archive,
            pytest.raises(tarfile.OutsideDestinationError),
        ):
            extract(archive, str(destination))
        assert not (destination.parent / "destevil/payload.txt").exists()
        with _tarball("ok.txt", legacy_default=name == "legacy") as archive:
            extract(archive, str(destination))
        assert (destination / "ok.txt").read_bytes() == b"controlled marker"
    sibling = tmp_path / "legacy/destevil"
    sibling.mkdir()
    with _tarball("redirect/payload.txt", legacy_default=True, symlink=True) as archive:
        _safe_extract(source, legacy)(archive, str(tmp_path / "legacy/dest"))
    assert (sibling / "payload.txt").read_bytes() == b"controlled marker"
