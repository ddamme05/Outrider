"""Reconstruct a faithful multi-file PR diff from a local git range.

Backs the demo's #6 "Outrider reviewing Outrider" showcase
(`specs/2026-06-21-demo-deployment.md`): a real ~27-file feature-arc diff fed
through the live graph, demonstrating dashboard-at-scale + triage cost-control.

`START..END` -> one `FileEntry` per changed file, in the EXACT shape the intake
node expects. The shape was verified against `src/outrider/agent/nodes/intake.py`
+ `src/outrider/schemas/pr_context.py` (the demo-build-contracts workflow):

  - `status`            : added | modified | removed | renamed
  - `patch`             : hunks-only, NO `--- a/` / `+++ b/` headers (GitHub's
                          `/pulls/{n}/files` wire shape; headers are re-synthesized
                          downstream by coordinates._wrap_github_hunks_with_headers)
  - `content_base`      : base-side text (None for `added`)
  - `content_head`      : head-side text (None for `removed`)
  - `previous_path`     : the old path, for `renamed` only
  - `additions`/`deletions` : counted from the unified diff itself (how GitHub
                          derives them), so the counts and the served patch agree

It also prints an HONEST offline dry-run summary. Triage tiers are an LLM (Haiku)
call (verified against `triage.py`), so this module CANNOT predict
DEEP/STANDARD/SKIM and deliberately refuses to claim a tier. It reports only what
is knowable offline: per-file path/status/±lines/bytes/estimated-tokens, the
configured analyze budget, the per-file cap, the high-risk reserve, the
starvation threshold, and a clearly-labelled (ceiling, not prediction) pressure
heuristic.

This module is pure local tooling: it shells out to `git` with argv lists (never
`shell=True`), validates both refs before use, and reads only the local repo. It
performs no network or LLM call. The live seed run that consumes these entries is
a separate, paid step (`scripts/live_claude_smoke.py`).
"""

from __future__ import annotations

import math
import os
import re
import subprocess  # noqa: S404 — local dev tooling; argv lists only, never shell=True
from dataclasses import dataclass
from pathlib import Path

# Verified against analyze.py via the demo-build-contracts workflow:
#   DEFAULT_REVIEW_BUDGET_TOKENS = 200_000 (env: OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS)
#   per-file cap = min(budget * 0.25, 60_000) ; high-risk reserve = budget * 0.25
#   COST_BUDGET_STARVATION fires at >= 3 COST_BUDGET_EXHAUSTED skips in a pass
#   token estimate = ceil(bytes / 3)
_DEFAULT_BUDGET_TOKENS = 200_000
_BUDGET_ENV = "OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS"
_PER_FILE_CAP_FRACTION = 0.25
_MAX_PER_FILE_TOKENS_ABSOLUTE = 60_000
_HIGH_RISK_RESERVE_FRACTION = 0.25
_STARVATION_THRESHOLD = 3
_BYTES_PER_TOKEN = 3

# Two-dot net diff only: `START..END`. Three-dot (`...`, merge-base symmetric
# diff) is deliberately rejected — this tool always runs `git diff START..END`,
# so accepting `...` and silently doing a two-dot diff would be a footgun. Refs may
# contain single dots (`v1.2.3`) but git forbids `..` inside a ref and forbids a
# leading/trailing dot, so anchoring each side to a non-dot boundary makes the `..`
# separator unambiguous. git rev-parse validates the refs after this shape check;
# the regex only rejects shell noise before anything reaches a subprocess argv.
_RANGE_RE = re.compile(
    r"^(?P<start>[A-Za-z0-9._/^~-]*[A-Za-z0-9_/^~-])\.\.(?P<end>[A-Za-z0-9_/^~-][A-Za-z0-9._/^~-]*)$"
)


class GitRangeError(RuntimeError):
    """A range/ref could not be validated or a git command failed."""


@dataclass(frozen=True)
class FileEntry:
    """One changed file, in intake's expected per-file shape."""

    path: str
    status: str  # added | modified | removed | renamed
    additions: int
    deletions: int
    patch: str | None  # hunks-only (no ---/+++ headers); None when there is no text hunk
    content_base: str | None  # None for added
    content_head: str | None  # None for removed
    previous_path: str | None  # set only for renamed

    @property
    def is_python(self) -> bool:
        return self.path.endswith(".py")

    @property
    def head_bytes(self) -> int:
        """Head-side size (base-side for a deletion) — the analyze input ceiling."""
        text = self.content_head if self.content_head is not None else self.content_base
        return len(text.encode("utf-8")) if text else 0

    @property
    def changed_bytes(self) -> int:
        """Byte size of the added+removed lines in the patch — a proxy for the
        changed scope analyze actually reviews (closer to real cost than head_bytes
        for a big file with a tiny diff)."""
        if self.patch is None:
            return 0
        total = 0
        for line in self.patch.splitlines():
            if (line.startswith("+") and not line.startswith("+++")) or (
                line.startswith("-") and not line.startswith("---")
            ):
                total += len(line[1:].encode("utf-8"))
        return total

    @property
    def estimated_tokens_ceiling(self) -> int:
        """ceil(bytes/3) over the WHOLE file — a loose UPPER bound, NOT a tier or
        a per-file cost. Analyze reviews only changed scope units, and triage may
        SKIM/SKIP the file entirely; both shrink the real number."""
        return math.ceil(self.head_bytes / _BYTES_PER_TOKEN)

    @property
    def estimated_tokens_floor(self) -> int:
        """ceil(changed_bytes/3) — a loose LOWER bound. Real analyze cost sits
        between this and the ceiling: it adds the enclosing scope + callers/callees
        + imports as context, but only for files triage routes to DEEP/STANDARD."""
        return math.ceil(self.changed_bytes / _BYTES_PER_TOKEN)


def _run_git(args: list[str], repo_root: Path) -> str:
    argv = ["git", "-C", str(repo_root), *args]  # git on PATH; argv list, never shell=True
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise GitRangeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _validate_ref(ref: str, repo_root: Path) -> None:
    """Reject anything git can't resolve to a commit, before it's used in a diff."""
    argv = ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise GitRangeError(f"not a valid commit-ish: {ref!r}")


def _hunks_only(patch_text: str) -> str | None:
    """Drop everything before the first `@@` hunk header.

    `git diff` emits `diff --git`, `index`, `--- a/`, `+++ b/` (and rename/similarity)
    lines before the hunks; GitHub's `/pulls/{n}/files` `patch` omits all of them.
    A pure rename / mode change with no content delta has no `@@` at all -> None,
    matching GitHub returning an empty/omitted patch for that case.
    """
    idx = patch_text.find("\n@@")
    if patch_text.startswith("@@"):
        body = patch_text
    elif idx != -1:
        body = patch_text[idx + 1 :]
    else:
        return None
    return body if body.strip() else None


def _count_changes(patch: str | None) -> tuple[int, int]:
    """Additions/deletions counted from the unified diff — how GitHub derives them."""
    if patch is None:
        return (0, 0)
    additions = deletions = 0
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return (additions, deletions)


def _show(ref: str, path: str, repo_root: Path) -> str | None:
    """`git show ref:path` -> text, or None if the path doesn't exist at ref."""
    argv = ["git", "-C", str(repo_root), "show", f"{ref}:{path}"]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    return proc.stdout if proc.returncode == 0 else None


def _parse_range(range_spec: str) -> tuple[str, str]:
    """Split `START..END` (two-dot net diff only). See `_RANGE_RE`."""
    m = _RANGE_RE.match(range_spec.strip())
    if m is None:
        raise GitRangeError(
            f"range must look like START..END (got {range_spec!r}); two-dot net diff "
            "only, e.g. 0c70d18^..39c538b (three-dot '...' is not supported)"
        )
    return (m.group("start"), m.group("end"))


def build_file_entries_from_range(range_spec: str, repo_root: Path) -> list[FileEntry]:
    """Reconstruct one `FileEntry` per changed file in `START..END`.

    Validates both refs, then for each name-status entry derives the hunks-only
    patch, the add/del counts (from the patch), and the base/head content per
    status. Ordering follows `git diff --name-status` (stable, deterministic).
    """
    start, end = _parse_range(range_spec)
    _validate_ref(start, repo_root)
    _validate_ref(end, repo_root)
    diff_range = f"{start}..{end}"

    name_status = _run_git(["diff", "--name-status", "-M", diff_range], repo_root)
    entries: list[FileEntry] = []
    for raw in name_status.splitlines():
        if not raw.strip():
            continue
        cols = raw.split("\t")
        code = cols[0]
        if code.startswith("R"):
            status, prev_path, path = "renamed", cols[1], cols[2]
            patch = _hunks_only(
                _run_git(["diff", "-M", diff_range, "--", prev_path, path], repo_root)
            )
            content_base = _show(start, prev_path, repo_root)
            content_head = _show(end, path, repo_root)
        elif code.startswith("C"):  # copy: treat as an added file at the new path
            status, prev_path, path = "added", None, cols[2]
            patch = _hunks_only(_run_git(["diff", diff_range, "--", path], repo_root))
            content_base, content_head = None, _show(end, path, repo_root)
        else:
            path = cols[1]
            prev_path = None
            patch = _hunks_only(_run_git(["diff", diff_range, "--", path], repo_root))
            if code.startswith("A"):
                status, content_base, content_head = "added", None, _show(end, path, repo_root)
            elif code.startswith("D"):
                status, content_base, content_head = "removed", _show(start, path, repo_root), None
            else:  # M (and any T/typechange falls through to modified)
                status = "modified"
                content_base = _show(start, path, repo_root)
                content_head = _show(end, path, repo_root)
        additions, deletions = _count_changes(patch)
        entries.append(
            FileEntry(
                path=path,
                status=status,
                additions=additions,
                deletions=deletions,
                patch=patch,
                content_base=content_base,
                content_head=content_head,
                previous_path=prev_path,
            )
        )
    return entries


def _budget_tokens() -> int:
    raw = os.environ.get(_BUDGET_ENV, "")
    if raw.strip().isdigit():
        return int(raw)
    return _DEFAULT_BUDGET_TOKENS


def summarize_dry_run(entries: list[FileEntry], range_spec: str) -> str:
    """An HONEST offline summary — size/status/budget only, never a predicted tier.

    Triage (an LLM call) decides DEEP/STANDARD/SKIM and analyze reviews only
    changed scope units, so this reports CEILINGS and refuses to claim which files
    will be reviewed or what will be found.
    """
    budget = _budget_tokens()
    per_file_cap = min(int(budget * _PER_FILE_CAP_FRACTION), _MAX_PER_FILE_TOKENS_ABSOLUTE)
    reserve = int(budget * _HIGH_RISK_RESERVE_FRACTION)

    py = [e for e in entries if e.is_python]
    nonpy = [e for e in entries if not e.is_python]
    ceiling = sum(e.estimated_tokens_ceiling for e in py)
    floor = sum(e.estimated_tokens_floor for e in py)
    total_add = sum(e.additions for e in entries)
    total_del = sum(e.deletions for e in entries)
    status_counts: dict[str, int] = {}
    for e in entries:
        status_counts[e.status] = status_counts.get(e.status, 0) + 1

    lines: list[str] = []
    lines.append(f"  Dry-run reconstruction of {range_spec}  (OFFLINE — no LLM, no network)")
    lines.append(f"    files .............. {len(entries)}  ({_fmt_counts(status_counts)})")
    lines.append(f"    line delta ......... +{total_add} / -{total_del}")
    lines.append(
        f"    python (analyzable)  {len(py)}   non-python (analyze skips, V1) {len(nonpy)}"
    )
    lines.append(f"    budget ............. {budget:,} tokens  ({_BUDGET_ENV})")
    lines.append(
        f"    per-file cap ....... {per_file_cap:,} tokens   high-risk reserve {reserve:,}"
    )
    lines.append(
        f"    starvation ......... anomaly at >= {_STARVATION_THRESHOLD} "
        "COST_BUDGET_EXHAUSTED skips"
    )
    lines.append("")
    header = f"    {'path':<52} {'status':<9} {'+':>5} {'-':>5} {'~bytes':>8} {'~tok':>7}  note"
    lines.append(header)
    for e in sorted(entries, key=lambda x: x.estimated_tokens_ceiling, reverse=True):
        note = "" if e.is_python else "non-python: analyze skips (V1)"
        lines.append(
            f"    {e.path:<52} {e.status:<9} {e.additions:>5} {e.deletions:>5} "
            f"{e.head_bytes:>8} {e.estimated_tokens_ceiling:>7}  {note}"
        )
    lines.append("")
    lines.append(
        f"    python analyze est-tokens (loose bounds) .. floor ~{floor:,} (changed content) .. "
        f"ceiling ~{ceiling:,} (whole-file)"
    )
    lines.append(f"    budget {budget:,} sits inside these bounds — NOT a pressure verdict:")
    lines.append("    real cost = changed scope units + context, only for files triage routes to")
    lines.append(
        "    DEEP/STANDARD. The ceiling is inflated by big files with tiny diffs (e.g. a 150KB"
    )
    lines.append(
        "    file with +8 lines reviews ~8 lines, not 50K tokens). Triage (an LLM call) decides"
    )
    lines.append(
        "    tiers — NOT predicted here. This run claims no tier, no per-file cost, no finding;"
    )
    lines.append("    the seed-capture check confirms admission/starvation against the real review")
    lines.append("    (specs/2026-06-21-demo-deployment.md).")
    return "\n".join(lines)


def _fmt_counts(counts: dict[str, int]) -> str:
    order = ["added", "modified", "removed", "renamed"]
    parts = [f"{counts[s]} {s}" for s in order if s in counts]
    return ", ".join(parts) if parts else "none"


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Dry-run reconstruct a git range for the demo seed."
    )
    parser.add_argument("range", help="git range START..END (e.g. 0c70d18^..39c538b)")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repo root (defaults to this script's repo)",
    )
    args = parser.parse_args()
    try:
        entries = build_file_entries_from_range(args.range, args.repo_root)
    except GitRangeError as exc:
        print(f"  error: {exc}", flush=True)
        return 2
    print(summarize_dry_run(entries, args.range), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
