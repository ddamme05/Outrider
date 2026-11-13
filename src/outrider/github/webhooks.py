"""Verify GitHub webhook signatures before parsing request data.

The wrapper delegates to the GitHub SDK's constant-time HMAC comparison.
Callers must pass the unmodified request bytes and selected webhook secret.

See webhook-signature-constant-time-compare and docs/trust-boundaries.md.
"""

from githubkit.webhooks import verify

__all__ = ["verify_webhook_signature"]


def verify_webhook_signature(
    secret: str,
    body: bytes,
    signature_header: str,
) -> bool:
    """Verify a GitHub webhook signature against the raw request body.

    Args:
        secret: The webhook signing secret as raw text (caller's
            responsibility to `.get_secret_value()` from the SecretStr
            settings field at the call site).
        body: The raw request body bytes captured BEFORE any JSON
            parsing — re-serializing a parsed payload would produce
            different bytes and fail verification (and worse, mask
            content/HMAC mismatch attacks).
        signature_header: The `X-Hub-Signature-256` header value
            exactly as received, including the `sha256=` prefix. No
            pre-normalization here.

    Returns:
        `True` iff the signature is valid for the given body+secret.
        `False` for any signature mismatch — including malformed digest,
        wrong-length header, base64 garbage, or HMAC mismatch.
        `githubkit.webhooks.verify` does NOT raise on mismatch; it
        returns False (its implementation calls `hmac.compare_digest`
        unconditionally — see `githubkit.versions.*.webhooks._namespace`).

    Raises:
        Programming errors only (e.g., `AttributeError` if `signature_header`
        is not a `str`; `TypeError` if `body` is not `bytes`). The router
        does NOT collapse these to 401 — they propagate as 5xx so
        unexpected verifier faults (dependency regressions, wrong-shape
        inputs) are operator-visible rather than masquerading as auth
        failures. The route returns 401 ONLY on `False` from this
        function.
    """
    return verify(secret, body, signature_header)
