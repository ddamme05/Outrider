# See DECISIONS.md#070 — the GitHub App Manifest builder + the bound contract.
"""The GitHub App Manifest builder (`DECISIONS.md#070`).

`POST /setup` returns a manifest that the operator's browser auto-submits to
`github.com/organizations/{org}/settings/apps/new?state=<signed-state>`; GitHub creates the App and
redirects to `redirect_url?code=&state=`. Everything GitHub needs is declared here, and ALL URLs are
built from `OUTRIDER_PUBLIC_BASE_URL` — never the request `Host` header (an attacker-controlled
header must not steer where GitHub sends the code/credentials).

The manifest declares `public: false` (so the App installs only on its owning org, keeping V1's
one-org model), `default_permissions {contents: read, pull_requests: write}`, and `default_events
[pull_request]`. (`installation` + `installation_repositories` are auto-delivered to every App and
cannot be subscribed to, so they are NOT declared — the App still receives them.) `build_manifest`
also returns the `manifest_contract_digest` — a hash of the exact manifest THIS attempt sent,
recorded on `setup_state`. The single-use **nonce** binds the callback to the attempt; the digest is
a **deployment-continuity guard** — the callback re-derives it from the stored org + current config
and rejects on mismatch (`router._verify_attempt_digest`), catching an `OUTRIDER_PUBLIC_BASE_URL`
change between Start and callback that would leave the App with URLs no longer pointing here.

The **response-verifiable** expectations (`EXPECTED_PERMISSIONS` / `EXPECTED_EVENTS`) are what the
response must match (`binding.verify_conversion_binding`): permissions is the declared map PLUS
GitHub's implicit `metadata: read`; events is the same set the manifest declared (only the
subscribable ones). `public` is submission-only — the response omits it, so it is not
response-verified.
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
    """
    manifest: dict[str, object] = {
        "name": name,
        "url": base_url,
        "hook_attributes": {"url": f"{base_url}/webhooks/github"},
        "redirect_url": f"{base_url}/setup/callback",
        "public": False,
        "default_permissions": dict(MANIFEST_PERMISSIONS),
        "default_events": list(MANIFEST_EVENTS),
    }
    digest = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest, digest
