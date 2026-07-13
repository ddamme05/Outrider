"""Unit tests for `api/setup/binding` — conversion-response binding verification (#070)."""

from __future__ import annotations

import pytest

from outrider.api.setup.binding import BindingMismatchError, verify_conversion_binding
from outrider.api.setup.manifest import EXPECTED_EVENTS, EXPECTED_PERMISSIONS
from outrider.api.setup.state_machine import SetupBinding

# GitHub's ACTUAL conversion-response wire shape — NOT an echo of our constants: the `events` array
# holds only the SUBSCRIBABLE events (installation/installation_repositories are auto-delivered, not
# echoed), and `permissions` includes GitHub's implicit `metadata:read`.
_GOOD_PERMS = {"metadata": "read", "contents": "read", "pull_requests": "write"}
_GOOD_EVENTS = ["pull_request"]


def _binding(
    *,
    org: str | None = "acme",
    perms: dict[str, str] | None = None,
    events: list[str] | None = None,
) -> SetupBinding:
    return SetupBinding(
        expected_org_login=org,
        expected_permissions=dict(EXPECTED_PERMISSIONS) if perms is None else perms,
        expected_events=list(EXPECTED_EVENTS) if events is None else events,
        manifest_contract_digest="digest",
    )


def test_matching_binding_passes() -> None:
    verify_conversion_binding(
        owner_login="acme", permissions=_GOOD_PERMS, events=_GOOD_EVENTS, binding=_binding()
    )


def test_realistic_github_wire_shape_accepted() -> None:
    """Regression against the events-binding bug: the manifest constants (`EXPECTED_PERMISSIONS` /
    `EXPECTED_EVENTS`, which the binding is built from) must MATCH GitHub's actual conversion wire
    shape — subscribable-only events + implicit `metadata:read`. Fails if `EXPECTED_EVENTS` ever
    re-lists the non-subscribable `installation` events GitHub strips from the response."""
    verify_conversion_binding(
        owner_login="acme",
        permissions={"metadata": "read", "contents": "read", "pull_requests": "write"},
        events=["pull_request"],
        binding=_binding(),  # expected_* = the manifest constants
    )


def test_owner_is_case_insensitive() -> None:
    verify_conversion_binding(
        owner_login="Acme",
        permissions=_GOOD_PERMS,
        events=_GOOD_EVENTS,
        binding=_binding(org="acme"),
    )


def test_owner_mismatch_rejected() -> None:
    with pytest.raises(BindingMismatchError, match="owner"):
        verify_conversion_binding(
            owner_login="attacker", permissions=_GOOD_PERMS, events=_GOOD_EVENTS, binding=_binding()
        )


def test_wider_permissions_rejected() -> None:
    """The core defense: a stolen state + swapped code for a wider-permission App is rejected."""
    wider = {**_GOOD_PERMS, "administration": "write"}
    with pytest.raises(BindingMismatchError, match="permissions"):
        verify_conversion_binding(
            owner_login="acme", permissions=wider, events=_GOOD_EVENTS, binding=_binding()
        )


def test_events_are_order_independent() -> None:
    verify_conversion_binding(
        owner_login="acme",
        permissions=_GOOD_PERMS,
        events=list(reversed(_GOOD_EVENTS)),
        binding=_binding(),
    )


def test_events_mismatch_rejected() -> None:
    with pytest.raises(BindingMismatchError, match="events"):
        verify_conversion_binding(
            owner_login="acme", permissions=_GOOD_PERMS, events=["push"], binding=_binding()
        )


def test_none_expectation_fails_closed() -> None:
    empty = SetupBinding(
        expected_org_login=None,
        expected_permissions=None,
        expected_events=None,
        manifest_contract_digest=None,
    )
    with pytest.raises(BindingMismatchError):
        verify_conversion_binding(
            owner_login="acme", permissions=_GOOD_PERMS, events=_GOOD_EVENTS, binding=empty
        )
