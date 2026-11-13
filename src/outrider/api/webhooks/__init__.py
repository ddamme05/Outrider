"""Receive signed GitHub deliveries at POST /webhooks/github.

router.py handles the request, schemas.py parses the fields the app needs,
and signature.py verifies the delivery through the GitHub wrapper. SDK imports
stay in their wrappers (vendor-sdks-only-in-wrappers).

See docs/trust-boundaries.md#5-input-boundary.
"""
