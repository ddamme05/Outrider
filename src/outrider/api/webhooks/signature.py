# Route-facing webhook signature verification per docs/trust-boundaries.md:87.
"""`verify_signature` — the only file performing webhook signature comparison.

Route-facing security-critical entry point preserved per
[docs/trust-boundaries.md:87](docs/trust-boundaries.md#L87) (*"the only
file that performs signature comparison; security-critical"*). Delegates
internally to `outrider.github.webhooks.verify_webhook_signature` (the
vendor wrapper).

The two-module path satisfies both:
  - `docs/trust-boundaries.md#5-input-boundary` — this module is the
    route-facing entry the doc names.
  - `vendor-sdks-only-in-wrappers` — the actual `githubkit` import lives
    in `github/webhooks.py`, not here.

`webhook-signature-constant-time-compare` is satisfied transitively:
the underlying `githubkit.webhooks.verify` uses `hmac.compare_digest`
per upstream docs (MCP-verified at `githubkit/usage/webhooks.md`).

No custom header pre-normalization (no `startswith`, `split`, `lower`,
`strip`) is performed here OR in the wrapper. The header passes through
to the verifier exactly as received from the route handler.
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
