"""Build the GitHub App manifest submitted by the setup page.

build_manifest returns the requested App configuration and a digest used to
check that deployment settings did not change during setup. URLs come from
OUTRIDER_PUBLIC_BASE_URL, never the request Host header. binding.py checks the
organization, permissions and events returned by GitHub.

See DECISIONS.md#070.
"""

from __future__ import annotations

import json
from hashlib import sha256
from types import MappingProxyType

__all__ = [
    "EXPECTED_EVENTS",
    "EXPECTED_PERMISSIONS",
    "MANIFEST_EVENTS",
    "MANIFEST_PERMISSIONS",
    "build_manifest",
]

# What the manifest DECLARES (sent to GitHub). Read-only mappings so a caller can't mutate the
# shared contract. Fine-grained permission names + access levels; minimum-viable scope (#070).
MANIFEST_PERMISSIONS: MappingProxyType[str, str] = MappingProxyType(
    {"contents": "read", "pull_requests": "write"}
)
# ONLY the subscribable event Outrider declares. `installation` + `installation_repositories` are
# NOT declared: per the GitHub webhook docs they are auto-delivered to every App ("You cannot
# manually subscribe to this event"), so GitHub strips them from `default_events` and never echoes
# them in the conversion response's `events` array — declaring them would false-orphan every real
# onboarding (the set-equality binding check would never match). The App still receives them.
MANIFEST_EVENTS: tuple[str, ...] = ("pull_request",)

# What the conversion RESPONSE must contain (response-verifiable). GitHub adds `metadata: read`
# implicitly to every App, so the expected permissions map is the declared map plus metadata; the
# expected events are the same set the manifest declared.
EXPECTED_PERMISSIONS: MappingProxyType[str, str] = MappingProxyType(
    {**MANIFEST_PERMISSIONS, "metadata": "read"}
)
EXPECTED_EVENTS: tuple[str, ...] = MANIFEST_EVENTS


def build_manifest(*, base_url: str, name: str) -> tuple[dict[str, object], str]:
    """Build the GitHub App Manifest for THIS attempt and its content digest.

    `base_url` is the canonical `OUTRIDER_PUBLIC_BASE_URL` (already validated + trailing-slash
    stripped); every URL derives from it. `name` is the operator-editable default App name. Returns
    `(manifest, digest)` where `digest = sha256(canonical-json)` — stored on `setup_state` for the
    callback's deployment-continuity check (`router._verify_attempt_digest`), NOT the callback-to-
    attempt binding (the single-use nonce is that).

    **The digest folds EVERY key**, so changing this dict rotates it. A deploy that lands while an
    attempt sits in `AWAITING_CALLBACK` makes that attempt fail `_verify_attempt_digest` and orphan
    — recoverable via `POST /setup/reset` then a fresh Start, but operators should drain in-flight
    attempts across a manifest change. Existing Apps do NOT inherit manifest edits; a changed key
    applies to Apps created from a NEW manifest, or must be set by hand in the App's settings.

    `redirect_url` and `setup_url` are DIFFERENT hops and both are required:
      - `redirect_url` (`/setup/callback`) receives GitHub's one-time manifest `code`, and its
        handler redirects onward to GitHub's install screen. That hop stays as-is.
      - `setup_url` (`/setup`) is where GitHub sends the operator AFTER they install the App,
        with `?installation_id=…&setup_action=install`. Without it the browser is stranded on
        GitHub and the operator has to find their way back to Outrider by hand.
    `setup_on_update` is deliberately OMITTED: with it set, GitHub also redirects here whenever an
    operator merely edits the App's repository access, which would yank them out of the settings
    page they were working in. Post-install is the only moment this redirect is wanted.

    The `installation_id` GitHub appends to `setup_url` is UNTRUSTED — GitHub's own docs warn it
    can be spoofed, and this endpoint is public. It is a HINT only ("an install probably just
    happened, start polling"); `GET /setup/status` re-derives `install_known` from the local
    installations table, which is fed by the signature-verified installation webhook.
    """
    manifest: dict[str, object] = {
        "name": name,
        "url": base_url,
        "hook_attributes": {"url": f"{base_url}/webhooks/github"},
        "redirect_url": f"{base_url}/setup/callback",
        # Post-install landing page. `GET /setup` has no API route — it falls through to the SPA
        # catch-all (`api/spa.py`, `/setup` is descendant-only reserved), so no backend route is
        # needed and the query string never enters the routing decision.
        "setup_url": f"{base_url}/setup",
        "public": False,
        "default_permissions": dict(MANIFEST_PERMISSIONS),
        "default_events": list(MANIFEST_EVENTS),
    }
    digest = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest, digest
