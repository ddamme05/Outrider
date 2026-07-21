"""Unit tests for `api/setup/manifest` — the GitHub App Manifest builder (#070)."""

from __future__ import annotations

from outrider.api.setup.manifest import (
    EXPECTED_EVENTS,
    EXPECTED_PERMISSIONS,
    MANIFEST_EVENTS,
    MANIFEST_PERMISSIONS,
    build_manifest,
)

_BASE = "https://ci.acme.com"


def test_manifest_structure() -> None:
    manifest, digest = build_manifest(base_url=_BASE, name="Outrider acme")
    assert manifest["name"] == "Outrider acme"
    assert manifest["url"] == _BASE
    assert manifest["hook_attributes"] == {"url": f"{_BASE}/webhooks/github"}
    assert manifest["redirect_url"] == f"{_BASE}/setup/callback"
    assert manifest["setup_url"] == f"{_BASE}/setup"
    assert manifest["public"] is False
    assert manifest["default_permissions"] == {"contents": "read", "pull_requests": "write"}
    assert manifest["default_events"] == list(MANIFEST_EVENTS)
    assert isinstance(digest, str) and len(digest) == 64  # sha256 hex


def test_manifest_key_set_is_exhaustively_pinned() -> None:
    """The digest folds EVERY key, so an added one silently rotates it — and a rotation
    that lands mid-attempt orphans any in-flight `AWAITING_CALLBACK`. The per-key
    assertions above pass happily when a key is ADDED, so pin the whole set: changing
    it should be a deliberate edit here, not a side effect noticed in production.
    """
    manifest, _ = build_manifest(base_url=_BASE, name="x")
    assert set(manifest) == {
        "name",
        "url",
        "hook_attributes",
        "redirect_url",
        "setup_url",
        "public",
        "default_permissions",
        "default_events",
    }


def test_setup_on_update_is_omitted() -> None:
    """With `setup_on_update` set, GitHub redirects to `setup_url` whenever an operator
    edits the App's repository access — yanking them out of the settings page they were
    working in. Post-install is the only moment that redirect is wanted."""
    manifest, _ = build_manifest(base_url=_BASE, name="x")
    assert "setup_on_update" not in manifest


def test_urls_derive_from_base_not_host() -> None:
    manifest, _ = build_manifest(base_url="https://other.example", name="x")
    assert manifest["hook_attributes"]["url"] == "https://other.example/webhooks/github"
    assert manifest["redirect_url"] == "https://other.example/setup/callback"
    assert manifest["setup_url"] == "https://other.example/setup"


def test_digest_is_stable_and_sensitive() -> None:
    d1 = build_manifest(base_url=_BASE, name="A")[1]
    d2 = build_manifest(base_url=_BASE, name="A")[1]
    d3 = build_manifest(base_url=_BASE, name="B")[1]
    assert d1 == d2  # deterministic — a restart-spanning callback verifies against the same digest
    assert d1 != d3  # name change → different digest
    assert (
        build_manifest(base_url="https://x.example", name="A")[1] != d1
    )  # base change → different


def test_expected_permissions_include_implicit_metadata() -> None:
    # The manifest DECLARES contents+pull_requests; GitHub adds metadata:read, so the response-
    # verifiable expectation is the declared map PLUS metadata.
    assert dict(MANIFEST_PERMISSIONS) == {"contents": "read", "pull_requests": "write"}
    assert dict(EXPECTED_PERMISSIONS) == {
        "contents": "read",
        "pull_requests": "write",
        "metadata": "read",
    }
    assert tuple(EXPECTED_EVENTS) == MANIFEST_EVENTS
