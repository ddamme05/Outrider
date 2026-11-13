"""Check capture integrity with real PR metadata and local content envelopes."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from urllib.parse import parse_qs, unquote

import pytest

from outrider.coordinates import CoordinateError

from .capture_pr import capture_pull_request

_CAPTURE = Path(__file__).with_name("corpus") / "real_pr_refresh" / "starlette-985"


class CapturedGitHub:
    """Serve captured metadata and source; only the REST content envelope is synthetic."""

    def __init__(self) -> None:
        self.pull = json.loads((_CAPTURE / "raw/pr.json").read_text())
        self.files = json.loads((_CAPTURE / "raw/files.json").read_text())
        self.compare = json.loads((_CAPTURE / "raw/compare.json").read_text())
        receipt = json.loads((_CAPTURE / "receipt.json").read_text())
        fixture = json.loads((_CAPTURE / "fixture.json").read_text())
        self.repository = receipt["source"]["repo"]
        self.number = receipt["source"]["pr"]
        self.contents = {}
        for file, record in zip(fixture["files"], receipt["files"], strict=True):
            for side in ("base", "head"):
                if record[side] is not None:
                    path = file["previous_path"] or file["path"] if side == "base" else file["path"]
                    self.add_content(path, record[side], file[f"content_{side}"].encode("utf-8"))
        license_record = receipt["license"]
        self.add_content(
            license_record["path"], license_record, (_CAPTURE / "LICENSE").read_bytes()
        )
        self.pull_reads = 0
        self.move_head = False

    def add_content(self, path: str, record: dict, raw: bytes) -> None:
        self.contents[path, record["commit"]] = {
            "type": "file",
            "encoding": "base64",
            "size": len(raw),
            "content": base64.b64encode(raw).decode("ascii"),
            "sha": record["blob_sha"],
        }

    def __call__(self, endpoint: str) -> dict | list:
        pull_endpoint = f"repos/{self.repository}/pulls/{self.number}"
        if endpoint == pull_endpoint:
            self.pull_reads += 1
            value = copy.deepcopy(self.pull)
            if self.move_head and self.pull_reads > 1:
                value["head"]["sha"] = "0" * 40
            return value
        if endpoint == pull_endpoint + "/files?per_page=100":
            return self.files
        if "/compare/" in endpoint:
            return self.compare
        if "/contents/" in endpoint:
            path, query = endpoint.split("/contents/", 1)[1].split("?", 1)
            return self.contents[unquote(path), parse_qs(query)["ref"][0]]
        raise AssertionError(f"unexpected GitHub request: {endpoint}")


def capture(api: CapturedGitHub, output: Path) -> dict:
    return capture_pull_request(
        api.repository,
        api.number,
        output,
        license_path="LICENSE.md",
        fetch_json=api,
    )


def test_real_capture_round_trip_retains_source_license_and_unscripted_inputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "capture"
    receipt = capture(CapturedGitHub(), output)
    for name in ("fixture.json", "LICENSE", "raw/pr.json", "raw/files.json", "raw/compare.json"):
        assert (output / name).read_bytes() == (_CAPTURE / name).read_bytes()
    assert json.loads((output / "fixture.json").read_text())["llm_responses"] == {}
    assert receipt["raw_files"] == json.loads((_CAPTURE / "receipt.json").read_text())["raw_files"]
    with pytest.raises(ValueError, match="already exists"):
        capture(CapturedGitHub(), output)


@pytest.mark.parametrize(
    "fault",
    ["missing-file", "moving-head", "altered-source", "truncated-patch", "wrong-patch-text"],
)
def test_incomplete_or_mixed_inputs_never_create_a_capture(tmp_path: Path, fault: str) -> None:
    api = CapturedGitHub()
    if fault == "missing-file":
        api.files = []
        message = "changed-file list is incomplete"
    elif fault == "moving-head":
        api.move_head = True
        message = "PR changed during capture"
    elif fault == "truncated-patch":
        api.files[0]["patch"] = api.files[0]["patch"].rsplit("\n@@", 1)[0]
        message = "patch totals"
    elif fault == "wrong-patch-text":
        lines = api.files[0]["patch"].splitlines(keepends=True)
        index = next(index for index, line in enumerate(lines) if line.startswith("+"))
        lines[index] = "+THIS_IS_NOT_THE_CAPTURED_HEAD_SOURCE\n"
        api.files[0]["patch"] = "".join(lines)
        message = "patch does not reconstruct"
    else:
        content = next(iter(api.contents.values()))
        content["content"] = base64.b64encode(b" " * content["size"]).decode("ascii")
        message = "Git blob identity"
    output = tmp_path / "capture"
    with pytest.raises(ValueError, match=message):
        capture(api, output)
    assert not output.exists()


def test_upstream_paths_cannot_choose_a_local_output_location(tmp_path: Path) -> None:
    api = CapturedGitHub()
    api.files[0]["filename"] = "../outside.py"
    output = tmp_path / "capture"
    with pytest.raises(CoordinateError):
        capture(api, output)
    assert not output.exists()
