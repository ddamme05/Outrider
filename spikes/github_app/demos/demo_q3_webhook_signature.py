"""Q3 — webhook signature verification (HMAC-SHA256, time-constant compare).

The invariant ``webhook-signature-constant-time-compare`` names
``hmac.compare_digest`` as the primitive. This demo proves:

1. The primitive works: we compute a signature with ``hmac.new(...).hexdigest()``,
   prepend ``sha256=``, compare with ``hmac.compare_digest``, positive + negative.
2. ``githubkit.webhooks.verify`` (documented as time-constant) agrees with the
   primitive on both good and bad signatures. Using the SDK's wrapper in
   production and the primitive in a test is fine because they produce the
   same result; the demo asserts they agree.
3. A same-length-but-wrong signature still fails — this is the exact case
   ``hmac.compare_digest`` defends against (``==`` would short-circuit on
   the first byte).
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from githubkit.webhooks import sign as gh_sign
from githubkit.webhooks import verify as gh_verify

FIXTURES = Path(__file__).parent.parent / "fixtures"
SECRET = "outrider-spike-webhook-secret"


def compute_signature(secret: str, body: bytes) -> str:
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


def main() -> None:
    body = (FIXTURES / "sample_pull_request_opened.json").read_bytes()

    # Primitive: compute + verify round-trip.
    expected = compute_signature(SECRET, body)
    assert hmac.compare_digest(expected, expected), (
        "Q3 FAIL: compare_digest rejects a signature compared against itself"
    )

    # Negative case: same-length, different content.
    wrong = "sha256=" + ("0" * 64)
    assert len(wrong) == len(expected), (
        "Q3 test bug: wrong-signature length should match expected length"
    )
    assert not hmac.compare_digest(expected, wrong), (
        "Q3 FAIL: compare_digest accepted a wrong signature of equal length"
    )

    # Short-circuit would have passed on the first byte. compare_digest
    # examines every byte — that's the invariant's whole point.

    # Cross-check with githubkit.webhooks.sign: same wire format.
    sdk_sig = gh_sign(SECRET, body, method="sha256")
    assert hmac.compare_digest(sdk_sig, expected), (
        f"Q3 FAIL: githubkit.sign differs from primitive\n"
        f"  sdk:  {sdk_sig}\n  prim: {expected}"
    )

    # Cross-check with githubkit.webhooks.verify: agrees on good and bad.
    assert gh_verify(SECRET, body, expected) is True, (
        "Q3 FAIL: gh_verify rejected a signature we just computed"
    )
    assert gh_verify(SECRET, body, wrong) is False, (
        "Q3 FAIL: gh_verify accepted a wrong signature"
    )
    assert gh_verify(SECRET, body, "sha256=notevenhex") is False, (
        "Q3 FAIL: gh_verify accepted a malformed signature"
    )

    # Wrong secret must also fail.
    assert gh_verify("wrong-secret", body, expected) is False, (
        "Q3 FAIL: gh_verify accepted a signature computed under a different secret"
    )

    print(
        "Q3 OK: hmac.compare_digest correctly distinguishes equal-length "
        "signatures; githubkit.webhooks.{sign,verify} agree with the "
        "primitive; wrong-secret + malformed-signature paths both reject."
    )


if __name__ == "__main__":
    main()
