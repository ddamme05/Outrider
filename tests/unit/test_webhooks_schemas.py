"""Webhook payload schema tests.

Confirms the raw-payload Pydantic models parse a representative
`pull_request` event shape, reject malformed shapes, and tolerate the
forward-compat additions GitHub routinely makes.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from outrider.api.webhooks.schemas import (
    PullRequestEventPayload,
)


def _valid_payload() -> dict[str, Any]:
    """A minimally-valid `pull_request.opened` payload."""
    return {
        "action": "opened",
        "pull_request": {
            "number": 42,
            "title": "Add a thing",
            "body": "This PR adds a thing.",
            "user": {"login": "alice", "id": 100},
            "head": {"sha": "a" * 40, "ref": "feat/thing"},
            "base": {"sha": "b" * 40, "ref": "main"},
            "additions": 10,
            "deletions": 2,
        },
        "repository": {
            "id": 999,
            "full_name": "acme/widgets",
            "name": "widgets",
            "owner": {"login": "acme", "id": 200},
        },
        "installation": {"id": 12345},
    }


def test_valid_payload_parses() -> None:
    """Happy path — every required field present, types correct."""
    payload = PullRequestEventPayload.model_validate(_valid_payload())
    assert payload.action == "opened"
    assert payload.pull_request.number == 42
    assert payload.repository.id == 999
    assert payload.installation.id == 12345


def test_unknown_top_level_field_is_ignored() -> None:
    """Forward-compat — GitHub adds top-level fields routinely.

    `extra="ignore"` means we don't break when GitHub introduces a new
    top-level key (e.g., `auto_merge`, `requested_reviewers`).
    """
    payload_dict = _valid_payload()
    payload_dict["future_field"] = "anything"
    payload_dict["another_new_thing"] = {"nested": "structure"}

    payload = PullRequestEventPayload.model_validate(payload_dict)
    assert payload.action == "opened"


def test_pr_body_none_allowed() -> None:
    """GitHub's `pull_request.body` is `string | null` (per
    `DECISIONS.md#020`); the schema must accept None."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["body"] = None
    payload = PullRequestEventPayload.model_validate(payload_dict)
    assert payload.pull_request.body is None


def test_missing_installation_fails() -> None:
    """GitHub-App-only — payloads without `installation` are PAT-shaped
    and must not parse."""
    payload_dict = _valid_payload()
    del payload_dict["installation"]
    with pytest.raises(ValidationError) as exc:
        PullRequestEventPayload.model_validate(payload_dict)
    assert "installation" in str(exc.value)


def test_zero_or_negative_installation_id_rejected() -> None:
    """`installation.id` must be a positive integer (`ge=1`)."""
    payload_dict = _valid_payload()
    payload_dict["installation"]["id"] = 0
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_pr_number_must_be_positive() -> None:
    """`pull_request.number` is `ge=1`."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["number"] = 0
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_negative_additions_rejected() -> None:
    """Counts are `ge=0`; GitHub never sends negative."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["additions"] = -1
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_short_sha_rejected() -> None:
    """`head.sha` and `base.sha` are 40-hex SHA-1 — `min_length=40`
    catches a truncated value at parse time."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["head"]["sha"] = "a" * 10
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_unknown_action_parses_then_router_no_ops() -> None:
    """A new-to-us action string parses cleanly at the schema level.

    `action` is `str` (NOT a closed Literal) per the spec's "signed but
    unsupported → 2xx no-op" rule — a closed Literal would 400 on a
    future GitHub-added action, causing GitHub to retry indefinitely.
    The router's `_PULL_REQUEST_ACTION_ALLOWLIST` filter handles the
    no-op routing; that test lives in `test_webhook_router.py`.
    """
    payload_dict = _valid_payload()
    payload_dict["action"] = "this_is_a_future_github_action"
    payload = PullRequestEventPayload.model_validate(payload_dict)
    assert payload.action == "this_is_a_future_github_action"


@pytest.mark.parametrize(
    "action",
    ["opened", "synchronize", "reopened", "closed", "ready_for_review"],
)
def test_recognized_actions_parse(action: str) -> None:
    """All actions we recognize parse; allowlist enforcement is at the
    router level (only opened/synchronize/reopened proceed; others
    2xx no-op)."""
    payload_dict = _valid_payload()
    payload_dict["action"] = action
    payload = PullRequestEventPayload.model_validate(payload_dict)
    assert payload.action == action


def test_login_empty_rejected() -> None:
    """`WebhookUser.login` has `min_length=1` — empty login bypasses
    downstream URL-segment construction and prompt-emission paths."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["user"]["login"] = ""
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_login_with_slash_rejected() -> None:
    """`WebhookUser.login` pattern rejects forward slashes — a slashed
    login flowing into `repos/{owner}/...` URL segments would silently
    escape the per-repo scope."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["user"]["login"] = "alice/bob"
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_login_with_shell_metachars_rejected() -> None:
    """`WebhookUser.login` pattern rejects shell-metacharacter chars —
    not because the login reaches a shell (it doesn't — vendor-sdks-only-
    in-wrappers + no subprocess) but because the same characters tend
    to indicate forged-or-replayed payloads."""
    for bad in ["alice;", "alice|bob", "alice$x", "alice`x`", "alice\nbob"]:
        payload_dict = _valid_payload()
        payload_dict["pull_request"]["user"]["login"] = bad
        with pytest.raises(ValidationError):
            PullRequestEventPayload.model_validate(payload_dict)


def test_login_too_long_rejected() -> None:
    """`WebhookUser.login` has `max_length=39` matching GitHub's own
    server-side limit. A 40+ char login indicates a forged payload."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["user"]["login"] = "a" * 40
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_pr_title_too_long_rejected() -> None:
    """`WebhookPullRequest.title` has `max_length=4096`. Without this
    cap, a multi-MB title floods the audit-table payload JSONB and the
    LLM prompt (triage + analyze both reference `PRContext.pr_title`)."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["title"] = "x" * 4097
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_pr_body_too_long_rejected() -> None:
    """`WebhookPullRequest.body` has `max_length=65536`. Same rationale
    as title: bounds the prompt + audit-table cost contribution."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["body"] = "y" * 65537
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_models_are_frozen() -> None:
    """The parsed payload is immutable downstream of webhook parsing."""
    payload = PullRequestEventPayload.model_validate(_valid_payload())
    with pytest.raises(ValidationError):
        # `frozen=True` raises ValidationError on assignment when
        # `validate_assignment=False` is the implicit default. Pydantic v2
        # raises ValidationError for frozen models on field assignment.
        payload.action = "closed"  # type: ignore[misc]
