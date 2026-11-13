"""Reserve review budget for files with high-risk added lines.

Text patterns scan patch additions without requiring a source parser. Matches
are deliberately conservative: a false positive can reserve budget, while the
normal analysis still decides whether a finding exists. More precise checks
are tracked in FUP-162.

See specs/2026-06-17-analyze-cost-fairness.md.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping


class RiskSignature(NamedTuple):
    """One high-risk signature: a compiled pattern + a human description."""

    pattern: re.Pattern[str]
    description: str


# Compiled once at import. Each pattern is anchored to a specific call/literal
# shape and is linear, matched against the concatenation of the patch's ADDED
# lines (see `scan_added_lines_for_risk`).
RISK_SIGNATURES: Final[Mapping[str, RiskSignature]] = MappingProxyType(
    {
        "os_system": RiskSignature(
            re.compile(r"\bos\.system\s*\("),
            "os.system(...) — shell command execution (command injection)",
        ),
        "subprocess_shell_true": RiskSignature(
            re.compile(r"shell\s*=\s*True\b"),
            "subprocess(..., shell=True) — shell injection surface",
        ),
        "unsafe_deserialization": RiskSignature(
            # `yaml.load(` matches even the safe `yaml.load(s, Loader=SafeLoader)`
            # form — an accepted FP for this recall reserve (it does NOT match the
            # common-safe `yaml.safe_load(`). Costs one reserved slot, never a miss.
            re.compile(r"\bpickle\.loads?\s*\(|\byaml\.load\s*\("),
            "pickle.load/loads or yaml.load(...) — unsafe deserialization (RCE)",
        ),
        "hardcoded_secret": RiskSignature(
            re.compile(
                r"sk_live_[A-Za-z0-9]"
                r"|\bAKIA[0-9A-Z]{16}\b"
                r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
            ),
            "hardcoded secret literal (live key / AWS access key / private key)",
        ),
        "tls_verify_disabled": RiskSignature(
            re.compile(r"\bverify\s*=\s*False\b"),
            "verify=False — TLS verification disabled (MITM)",
        ),
        "weak_hash": RiskSignature(
            re.compile(r"\bhashlib\.(?:md5|sha1)\s*\("),
            "hashlib.md5/sha1(...) — weak hash (esp. for password/auth material)",
        ),
    }
)


def _added_lines(patch: str | None) -> str:
    """Concatenated text of the patch's ADDED lines (the new side).

    Plain lexical filtering of a unified-diff body: keep lines starting with a
    single `+`, strip the leading `+`, and skip the `+++ ` file header, context
    (` `), removed (`-`), and hunk-header (`@@`) lines. GitHub's
    `/pulls/{n}/files` patch is hunks-only (no `+++`/`---` headers in practice),
    but the `+++ ` guard is kept for safety. `None` patch (binary/oversized
    diff) → `""`.

    NOT coordinate translation — no line numbers, spans, or cross-system mapping
    — so it does not belong in `coordinates/`.
    """
    if not patch:
        return ""
    return "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def scan_added_lines_for_risk(patch: str | None) -> frozenset[str]:
    """Return the `RISK_SIGNATURES` keys whose pattern matches the patch's added
    lines. Empty frozenset if `patch` is `None` or nothing matches.

    A non-empty result marks the file high-risk for the analyze cost gate: it is
    eligible to draw from the bounded reserved budget slice so it cannot be
    starved purely by iteration position. The keys are catalog names — stable,
    sortable, and safe to surface in audit/telemetry (no PR content).
    """
    added = _added_lines(patch)
    if not added:
        return frozenset()
    return frozenset(name for name, sig in RISK_SIGNATURES.items() if sig.pattern.search(added))


__all__ = ["RISK_SIGNATURES", "RiskSignature", "scan_added_lines_for_risk"]
