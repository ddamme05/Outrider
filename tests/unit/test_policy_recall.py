"""policy.recall — high-risk signature scan for the analyze cost-budget reserve.

Scans the ADDED lines of a unified-diff patch for blatant CRITICAL-class
signatures. A recall reserve tolerates false positives by design (a false match
costs one reserved analyze slot); the tests pin (a) that each planted signature
is caught on the added side, (b) that removed/context lines are NOT scanned, and
(c) that the documented FP behavior (a match inside a comment still fires) holds.
"""

from __future__ import annotations

import pytest

from outrider.policy.recall import RISK_SIGNATURES, scan_added_lines_for_risk


def _added(*lines: str) -> str:
    """Build a GitHub-style hunks-only patch (no +++/--- headers) whose added
    side is `lines`."""
    body = "\n".join(f"+{line}" for line in lines)
    return f"@@ -1,1 +1,{len(lines)} @@\n{body}"


@pytest.mark.parametrize(
    ("snippet", "expected_key"),
    [
        ('os.system(f"ping -c 1 {host}")', "os_system"),  # the smoke's CRITICAL command_injection
        ("subprocess.run(cmd, shell=True)", "subprocess_shell_true"),
        ("data = pickle.loads(blob)", "unsafe_deserialization"),  # the smoke's HIGH
        ("cfg = yaml.load(stream)", "unsafe_deserialization"),
        ('API_KEY = "sk_live_abc123XYZ"', "hardcoded_secret"),  # the smoke's HIGH secret
        ("aws = 'AKIAIOSFODNN7EXAMPLE'", "hardcoded_secret"),
        ("requests.get(url, verify=False)", "tls_verify_disabled"),
        ("h = hashlib.md5(password.encode())", "weak_hash"),
        ("h = hashlib.sha1(token)", "weak_hash"),
    ],
)
def test_each_signature_matches_on_added_line(snippet: str, expected_key: str) -> None:
    assert expected_key in scan_added_lines_for_risk(_added(snippet))


def test_none_patch_is_empty() -> None:
    assert scan_added_lines_for_risk(None) == frozenset()


def test_clean_patch_is_empty() -> None:
    patch = _added("def add(a, b):", "    return a + b")
    assert scan_added_lines_for_risk(patch) == frozenset()


def test_removed_line_is_not_scanned() -> None:
    # A REMOVED os.system (deleting risky code) must NOT trip the reserve — the
    # PR is making things safer, not introducing the sink.
    patch = "@@ -1,1 +1,1 @@\n-os.system(cmd)\n+subprocess.run(['ls'])"
    assert "os_system" not in scan_added_lines_for_risk(patch)


def test_context_line_is_not_scanned() -> None:
    # A pre-existing os.system on an unchanged context line is not what this PR
    # adds; added-only scoping means it does not trip the reserve.
    patch = "@@ -1,2 +1,2 @@\n os.system(legacy)\n+log.info('touched nearby')"
    assert "os_system" not in scan_added_lines_for_risk(patch)


def test_multiple_signatures_on_one_patch() -> None:
    patch = _added('os.system(f"rm {p}")', "requests.get(u, verify=False)")
    result = scan_added_lines_for_risk(patch)
    assert {"os_system", "tls_verify_disabled"} <= result


def test_plus_plus_plus_header_not_treated_as_added() -> None:
    # Defensive: a `+++ ` file header must not be scanned as an added line.
    patch = "+++ b/app/os.system_in_path.py\n@@ -1,1 +1,1 @@\n+x = 1"
    assert scan_added_lines_for_risk(patch) == frozenset()


def test_fp_tolerance_comment_still_matches() -> None:
    # Documented behavior: the lexical scan does not understand comments, so a
    # signature inside an added comment still fires. Acceptable for a recall
    # reserve (costs one reserved slot, never a missed dangerous file).
    patch = _added("# never call os.system( here")
    assert "os_system" in scan_added_lines_for_risk(patch)


def test_catalog_keys_are_stable_and_sortable() -> None:
    # The result is a frozenset of catalog names — deterministic and safe to
    # surface in audit/telemetry (no PR content).
    assert set(RISK_SIGNATURES) >= {
        "os_system",
        "subprocess_shell_true",
        "unsafe_deserialization",
        "hardcoded_secret",
        "tls_verify_disabled",
        "weak_hash",
    }
