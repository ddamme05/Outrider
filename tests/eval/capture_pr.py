"""Save a public PR's exact source and provenance without making model calls.

Requires the GitHub CLI. Captures contain no expected answers; review and add
those separately before admitting a case to a model comparison.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote

from outrider.agent.eval_driver import EvalFixture, EvalFixtureFile
from outrider.coordinates import (
    CoordinateError,
    added_line_byte_ranges,
    lookup_patched_file,
    validate_diff_path,
)

MAX_FILES = 50
MAX_FILE_BYTES = 1_000_000
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
_COMMIT = re.compile(r"[a-f0-9]{40}")


def github_json(endpoint: str) -> Any:
    """Read public GitHub data using the CLI's existing authentication."""
    executable = shutil.which("gh")
    if executable is None:
        raise ValueError("install the GitHub CLI before capturing a PR")
    if not endpoint.startswith("repos/"):
        raise ValueError("only repository API endpoints are supported")
    result = subprocess.run(  # noqa: S603 — fixed read-only command; arguments never use a shell
        [executable, "api", "--hostname", "github.com", "--method", "GET", endpoint],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"GitHub could not read {endpoint}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _pull_identity(pull: dict) -> tuple:
    return (
        pull["base"]["sha"],
        pull["head"]["sha"],
        pull["merge_commit_sha"],
        pull["changed_files"],
        pull["additions"],
        pull["deletions"],
        pull["title"],
        pull["body"],
        pull["updated_at"],
    )


def _validate_patch(entry: dict, before: str | None, after: str | None) -> None:
    """Require complete hunk counts and exact base-to-head reconstruction."""
    path = validate_diff_path(entry["filename"])
    previous = (
        validate_diff_path(entry["previous_filename"]) if entry["status"] == "renamed" else path
    )
    patch = entry.get("patch")
    if not patch:
        raise ValueError(f"{path}: GitHub omitted the text patch; capture is incomplete")
    parsed = lookup_patched_file(patch, path)
    if parsed is None or (parsed.added, parsed.removed) != (entry["additions"], entry["deletions"]):
        raise ValueError(f"{path}: patch totals disagree with the changed-file metadata")
    if after is not None:
        added_line_byte_ranges(parsed, after)
    executable = shutil.which("git")
    if executable is None:
        raise ValueError("install Git before capturing a PR")
    old_name = json.dumps(f"a/{previous}", ensure_ascii=False)
    new_name = json.dumps(f"b/{path}", ensure_ascii=False)
    headers = [f"diff --git {old_name} {new_name}\n"]
    if before is None:
        headers.append("new file mode 100644\n")
    elif after is None:
        headers.append("deleted file mode 100644\n")
    elif previous != path:
        headers.extend([f"rename from {json.dumps(previous)}\n", f"rename to {json.dumps(path)}\n"])
    headers.extend(
        [
            f"--- {old_name if before is not None else '/dev/null'}\n",
            f"+++ {new_name if after is not None else '/dev/null'}\n",
        ]
    )
    encoded_patch = ("".join(headers) + patch.rstrip("\n") + "\n").encode("utf-8")
    env = {
        name: os.environ[name]
        for name in ("PATH", "LANG", "LC_ALL", "TMPDIR")
        if name in os.environ
    }
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    with TemporaryDirectory(prefix="outrider-pr-capture-") as directory:
        root = Path(directory)
        if before is not None:
            base_path = root / previous
            base_path.parent.mkdir(parents=True, exist_ok=True)
            base_path.write_bytes(before.encode("utf-8"))
        result = subprocess.run(  # noqa: S603 — Git applies validated paths only in a disposable directory
            [executable, "apply", "--whitespace=nowarn", "-"],
            input=encoded_patch,
            capture_output=True,
            cwd=root,
            env=env,
            timeout=10,
            check=False,
        )
        target = root / path
        matches = (
            not target.exists()
            if after is None
            else target.is_file() and target.read_bytes() == after.encode("utf-8")
        )
        if result.returncode or not matches:
            raise ValueError(f"{path}: patch does not reconstruct the captured head from its base")


def capture_pull_request(
    repository: str,
    number: int,
    output: Path,
    *,
    license_path: str = "LICENSE",
    fetch_json=github_json,
) -> dict:
    """Capture every changed text file, refusing incomplete or moving inputs.

    Files are read at immutable commits. Original hunks must match GitHub's line
    counts and reconstruct the complete head from its base in a disposable Git
    apply check. Nothing is saved until a second PR identity check succeeds.
    """
    if not _REPOSITORY.fullmatch(repository) or number < 1:
        raise ValueError("use OWNER/REPOSITORY and a positive PR number")
    if output.exists():
        raise ValueError("the output already exists; use a new capture directory")
    license_path = validate_diff_path(license_path)
    endpoint = f"repos/{repository}/pulls/{number}"
    pull = fetch_json(endpoint)
    if pull["base"]["repo"]["private"]:
        raise ValueError("this importer accepts public repositories only")
    repository = pull["base"]["repo"]["full_name"]
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("invalid canonical repository name")
    endpoint = f"repos/{repository}/pulls/{number}"
    if pull["number"] != number or not 1 <= pull["changed_files"] <= MAX_FILES:
        raise ValueError(f"capture requires one to {MAX_FILES} changed files")
    target_base, head = pull["base"]["sha"], pull["head"]["sha"]
    if not all(_COMMIT.fullmatch(value) for value in (target_base, head)):
        raise ValueError("PR refs must resolve to full commit hashes")
    comparison = fetch_json(f"repos/{repository}/compare/{target_base}...{head}")
    base = comparison["merge_base_commit"]["sha"]
    if not _COMMIT.fullmatch(base) or base == head:
        raise ValueError("GitHub did not supply a distinct PR diff merge-base")
    changed = fetch_json(f"{endpoint}/files?per_page=100")
    if len(changed) != pull["changed_files"]:
        raise ValueError("GitHub's changed-file list is incomplete")

    def read_source(path: str, commit: str) -> tuple[str, dict]:
        path = validate_diff_path(path)
        api_path = f"repos/{repository}/contents/{quote(path, safe='/')}?ref={commit}"
        data = fetch_json(api_path)
        if data.get("type") != "file" or data.get("encoding") != "base64":
            raise ValueError(f"{path}: only complete regular text files are supported")
        if not 0 <= data["size"] <= MAX_FILE_BYTES:
            raise ValueError(f"{path}: file exceeds the capture size limit")
        raw = base64.b64decode("".join(data["content"].split()), validate=True)
        if len(raw) != data["size"] or b"\x00" in raw:
            raise ValueError(f"{path}: incomplete or binary content")
        blob = hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw,
            usedforsecurity=False,
        ).hexdigest()
        if blob != data["sha"]:
            raise ValueError(f"{path}: content disagrees with its Git blob identity")
        text = raw.decode("utf-8")
        return text, {
            "sha256": _sha(raw),
            "commit": commit,
            "url": f"https://github.com/{repository}/blob/{commit}/{quote(path, safe='/')}",
            "blob_sha": data["sha"],
        }

    files, source_records = [], []
    seen = set()
    for entry in changed:
        path = validate_diff_path(entry["filename"])
        status = entry["status"]
        if path in seen or status not in {"added", "removed", "modified", "renamed"}:
            raise ValueError("changed paths must be unique and have a supported GitHub status")
        seen.add(path)
        previous = validate_diff_path(entry["previous_filename"]) if status == "renamed" else None
        before, before_record = (
            (None, None) if status == "added" else read_source(previous or path, base)
        )
        after, after_record = (None, None) if status == "removed" else read_source(path, head)
        record = after_record if after_record is not None else before_record
        if record["blob_sha"] != entry["sha"]:
            raise ValueError(f"{path}: changed-file and source blob identities disagree")
        _validate_patch(entry, before, after)
        files.append(
            EvalFixtureFile(
                path=path,
                status=status,
                additions=entry["additions"],
                deletions=entry["deletions"],
                patch=entry["patch"],
                previous_path=previous,
                content_base=before,
                content_head=after,
            )
        )
        source_records.append(
            {
                "path": path,
                "status": status,
                "previous_path": previous,
                "base": before_record,
                "head": after_record,
            }
        )
    if (
        sum(file.additions for file in files) != pull["additions"]
        or sum(file.deletions for file in files) != pull["deletions"]
    ):
        raise ValueError("PR totals disagree with the complete changed-file list")
    license_text, license_record = read_source(license_path, head)
    owner, repo = repository.split("/")
    fixture = EvalFixture(
        installation_id=9740,
        owner=owner,
        repo=repo,
        pr_number=number,
        base_sha=base,
        head_sha=head,
        pr_title=pull["title"],
        pr_body=pull["body"],
        author=pull["user"]["login"],
        total_additions=pull["additions"],
        total_deletions=pull["deletions"],
        files=tuple(files),
        llm_responses={},
    )
    if _pull_identity(fetch_json(endpoint)) != _pull_identity(pull):
        raise ValueError("PR changed during capture; retry into a new directory")
    artifacts = {
        "fixture.json": _json_bytes(fixture.model_dump(mode="json")),
        "LICENSE": license_text.encode("utf-8"),
        "raw/pr.json": _json_bytes(pull),
        "raw/files.json": _json_bytes(changed),
        "raw/compare.json": _json_bytes(comparison),
    }
    receipt = {
        "schema": "pr-capture-v1",
        "source": {
            "repo": repository,
            "pr": number,
            "url": pull["html_url"],
            "base_sha": base,
            "target_base_sha": target_base,
            "head_sha": head,
            "merge_sha": pull["merge_commit_sha"],
            "created_at": pull["created_at"],
            "updated_at": pull["updated_at"],
            "merged_at": pull["merged_at"],
            "captured_at": datetime.now(UTC).isoformat(),
        },
        "fixture_sha256": _sha(artifacts["fixture.json"]),
        "license": {"path": license_path, **license_record},
        "files": source_records,
        "context_files": [],
        "raw_files": {name: _sha(data) for name, data in artifacts.items()},
        "notes": (
            "Parsed REST JSON, not HTTP wire capture. Complete UTF-8 source; "
            "no model answers or labels. Installation identity is an eval placeholder."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in {**artifacts, "receipt.json": _json_bytes(receipt)}.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--license-path", default="LICENSE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = capture_pull_request(
            args.repo,
            args.pr,
            args.output,
            license_path=args.license_path,
        )
    except (CoordinateError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Capture failed: {exc}\n")
    print(
        json.dumps(
            {"capture": str(args.output), "source": receipt["source"], "model_calls_made": 0},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
