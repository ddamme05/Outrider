"""Webhook receiver — POST /webhooks/github.

Sub-modules:
  - `schemas` — raw-payload Pydantic models for the GitHub `pull_request`
    event (the input-boundary trust translation).
  - `signature` — route-facing security-critical module that performs
    HMAC verification; preserves the trust-boundary attribution at
    [docs/trust-boundaries.md:87](docs/trust-boundaries.md#L87).
    Delegates internally to `outrider.github.webhooks` (the vendor wrapper).
  - `router` — FastAPI router (lands in a later milestone of the
    intake-and-webhook spec).

This module does NOT import `githubkit` directly — that's confined to
`outrider.github.*` per the `vendor-sdks-only-in-wrappers` invariant.
"""
