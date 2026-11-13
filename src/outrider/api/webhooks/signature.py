"""Verify a webhook signature before the router reads its payload.

verify_signature delegates to github/webhooks.py, where the vendor verifier
performs the constant-time comparison. Pass the signature header unchanged;
normalizing it here would change what gets verified.

See docs/trust-boundaries.md#5-input-boundary,
vendor-sdks-only-in-wrappers and webhook-signature-constant-time-compare.
"""

from outrider.github.webhooks import verify_webhook_signature

__all__ = ["verify_signature"]


def verify_signature(
    secret: str,
    body: bytes,
    signature_header: str,
) -> bool:
    """Verify a GitHub webhook signature against the raw request body.

    Args:
        secret: The webhook signing secret. Caller is responsible for
            `.get_secret_value()` on the `SecretStr` at the route handler
            (the unwrap should happen here at the call site, not earlier
            in the request lifecycle).
        body: Raw request body bytes captured BEFORE any JSON parsing.
        signature_header: The `X-Hub-Signature-256` header value exactly
            as received, including the `sha256=` prefix.

    Returns:
        `True` iff the signature is valid for the given body+secret;
        `False` for any mismatch — including malformed digest,
        wrong-length header, base64 garbage, or HMAC mismatch.
        `githubkit.webhooks.verify` returns False (not raises) for these
        cases; its implementation calls `hmac.compare_digest`
        unconditionally. Route handler converts `False` to HTTP 401.

    Raises:
        Programming errors only (e.g., `AttributeError` if `signature_header`
        is not a `str`; `TypeError` if `body` is not `bytes`). The route
        handler does NOT collapse these to 401 — they propagate as 5xx
        so unexpected verifier faults (dependency regressions, wrong-
        shape inputs) surface to operators rather than masquerading as
        authentication failures. See `outrider.github.webhooks.verify_webhook_signature`
        for the matching contract in the wrapper layer.
    """
    return verify_webhook_signature(secret, body, signature_header)
