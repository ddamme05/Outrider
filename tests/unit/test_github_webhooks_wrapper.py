# Tests for the github/webhooks.py vendor wrapper.
"""Confirm `verify_webhook_signature` is a faithful delegation to
`githubkit.webhooks.verify`, and that the wrapper is the only call site
of `githubkit.webhooks` in the codebase (per `vendor-sdks-only-in-wrappers`).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from outrider.github.webhooks import verify_webhook_signature


def _sign(secret: str, body: bytes) -> str:
    """Compute the canonical `X-Hub-Signature-256` header for a body+secret.

    Used by the tests as the verifier of last resort — independent of
    githubkit, so a wrapper bug can't make a test pass that shouldn't.
    """
    mac = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={mac}"


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b'{"action": "opened", "number": 42}',
        b"\x00\x01\x02 binary-safe bytes \xff\xfe",
        b"a" * 64 * 1024,  # 64 KiB body
    ],
)
def test_valid_signature_returns_true(body: bytes) -> None:
    """A signature computed with the same secret+body verifies."""
    secret = "test-secret-123"  # noqa: S105 — test fixture, not a credential
    header = _sign(secret, body)
    assert verify_webhook_signature(secret, body, header) is True


def test_wrong_secret_returns_false() -> None:
    """Signature computed under a different secret does not verify."""
    body = b'{"event": "test"}'
    header = _sign("real-secret", body)
    assert verify_webhook_signature("wrong-secret", body, header) is False


def test_mutated_body_returns_false() -> None:
    """Body tampered after signature is computed does not verify."""
    secret = "test-secret"  # noqa: S105 — test fixture, not a credential
    original = b'{"action": "opened"}'
    header = _sign(secret, original)
    tampered = original + b" "  # trailing whitespace
    assert verify_webhook_signature(secret, tampered, header) is False


def test_signature_only_call_site() -> None:
    """`github/webhooks.py` is the ONLY file importing `githubkit.webhooks`.

    Enforces `vendor-sdks-only-in-wrappers` for the webhook-verification
    surface specifically. A future refactor that adds `from
    githubkit.webhooks import verify` to a non-wrapper file would silently
    bypass the wrapper layer; this test catches that at CI time.
    """
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src" / "outrider"

    # Anchor to real Python import statements so docstring / comment
    # mentions of `from githubkit.webhooks import ...` don't trigger
    # false positives. `^\s*` covers indented re-imports (rare but
    # legal); `\b` on the bare-import form prevents matching
    # `githubkit.webhooks_extra`.
    import_pattern = r"^\s*(from\s+githubkit\.webhooks\s+import|import\s+githubkit\.webhooks\b)"

    # Use ripgrep if available (faster + ignores binary files / venv); fall
    # back to a manual scan for environments without rg.
    rg = shutil.which("rg")
    if rg is not None:
        result = subprocess.run(  # noqa: S603 — fixed args, absolute rg path
            [
                rg,
                "--type",
                "py",
                "-l",
                "--multiline",  # `^` with multiline anchors per line
                import_pattern,
                str(src_root),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        hits = [line for line in result.stdout.splitlines() if line.strip()]
    else:
        compiled = re.compile(import_pattern, re.MULTILINE)
        hits = []
        for py_file in src_root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if compiled.search(text):
                hits.append(str(py_file))

    expected = str(src_root / "github" / "webhooks.py")
    assert hits == [expected], (
        f"Expected `githubkit.webhooks` import only at {expected!r}; "
        f"found at {hits!r}. Move the call into the wrapper."
    )


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Sub-second relevance: regression check requires modern import system.",
)
def test_wrapper_returns_bool_not_truthy_string() -> None:
    """Be explicit: the wrapper returns a real `bool`, not a truthy str /
    None / int. Defends downstream callers from accidentally treating
    `"sha256=..."` (or any string) as a successful verification.
    """
    secret = "s"  # noqa: S105 — test sentinel for bool-type assertion
    body = b"b"
    header = _sign(secret, body)
    result = verify_webhook_signature(secret, body, header)
    assert isinstance(result, bool)
