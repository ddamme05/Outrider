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
    InstallationEventPayload,
    InstallationRepositoriesEventPayload,
    InstallationRepositoryRef,
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
            "draft": False,
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
    assert payload.pull_request.draft is False


def test_missing_draft_rejected() -> None:
    """`draft` is REQUIRED (fail-closed autorun gate): GitHub always sends it on
    pull_request events, so a payload omitting it is malformed and must fail validation,
    not silently default to ready and autorun what could be a draft PR."""
    payload_dict = _valid_payload()
    del payload_dict["pull_request"]["draft"]
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


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


def test_non_hex_sha_rejected() -> None:
    """`sha` is bounded lowercase hex. A non-hex char (e.g., a slash or
    capital) → ValidationError. Without the pattern, raw bytes could
    flow into URL segments + audit payloads via a forged payload."""
    for bad in ["a" * 39 + "/", "A" * 40, "g" * 40, "z" * 40]:
        payload_dict = _valid_payload()
        payload_dict["pull_request"]["head"]["sha"] = bad
        with pytest.raises(ValidationError):
            PullRequestEventPayload.model_validate(payload_dict)


def test_overlong_sha_rejected() -> None:
    """`sha` has `max_length=64` (SHA-256). 65+ chars → ValidationError."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["head"]["sha"] = "a" * 65
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_sha256_length_admitted() -> None:
    """`sha` admits 64-hex (SHA-256) for forward-compat with GitHub's
    object-format migration on some surfaces."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["head"]["sha"] = "a" * 64
    payload_dict["pull_request"]["base"]["sha"] = "b" * 64
    payload = PullRequestEventPayload.model_validate(payload_dict)
    assert len(payload.pull_request.head.sha) == 64


def test_sha1_length_admitted() -> None:
    """Positive-boundary: `sha` admits 40-hex (SHA-1) — GitHub's default
    today. Pins the lower-bound inclusive admission paired with the
    upper-bound 64-hex test above."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["head"]["sha"] = "a" * 40
    payload_dict["pull_request"]["base"]["sha"] = "b" * 40
    payload = PullRequestEventPayload.model_validate(payload_dict)
    assert len(payload.pull_request.head.sha) == 40
    assert len(payload.pull_request.base.sha) == 40


def test_ref_empty_rejected() -> None:
    """`PullRequestRef.ref` has `min_length=1` — empty ref bypasses
    downstream prompt + audit-payload paths."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["head"]["ref"] = ""
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_ref_too_long_rejected() -> None:
    """`ref` `max_length=255` covers any real branch / tag name. 256+
    indicates a forged payload that would flow into prompts at
    inflated cost."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["head"]["ref"] = "a" * 256
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_ref_with_shell_metachars_rejected() -> None:
    """`ref` pattern rejects shell-meta + whitespace + traversal-style
    characters. Catches forged payloads with control chars / newlines /
    backticks that would flow into log lines and prompts."""
    for bad in ["feat;rm", "feat`x`", "feat$x", "feat\nbar", "feat ~bar", "feat:bar"]:
        payload_dict = _valid_payload()
        payload_dict["pull_request"]["head"]["ref"] = bad
        with pytest.raises(ValidationError):
            PullRequestEventPayload.model_validate(payload_dict)


def test_ref_realistic_shapes_admitted() -> None:
    """Real GitHub ref shapes admit cleanly: branch names with `/`,
    tag names with `.`, version-like names with `+`, dependabot/-shapes
    with `-` and `_`. Pins the cascade carve-out: pattern is bounded
    enough to reject shell-meta but permissive enough for legitimate
    `git check-ref-format` shapes."""
    for good in [
        "main",
        "feat/foo-bar",
        "release/v1.2.3",
        "renovate/lock-file-maintenance",
        "v1.0.0+build.42",
        "user.name/topic_branch",
    ]:
        payload_dict = _valid_payload()
        payload_dict["pull_request"]["head"]["ref"] = good
        payload = PullRequestEventPayload.model_validate(payload_dict)
        assert payload.pull_request.head.ref == good


def test_full_name_too_long_rejected() -> None:
    """`RepositoryRef.full_name` has `max_length=200`. The pattern
    matched any owner/name with a slash; without max_length, a forged
    payload could submit a multi-MB full_name."""
    payload_dict = _valid_payload()
    payload_dict["repository"]["full_name"] = "a" * 100 + "/" + "b" * 100 + "c" * 50
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_name_too_long_rejected() -> None:
    """`RepositoryRef.name` has `max_length=100` matching GitHub's own
    server-side cap. An oversized name flows into URL segments and
    prompts."""
    payload_dict = _valid_payload()
    payload_dict["repository"]["name"] = "a" * 101
    with pytest.raises(ValidationError):
        PullRequestEventPayload.model_validate(payload_dict)


def test_full_name_and_name_at_caps_admitted() -> None:
    """Positive-boundary: `full_name` at 200 chars (the cap) and `name`
    at 100 chars (the cap) admit cleanly. Pins the inclusive boundary
    so a strict-less-than regression flips this test."""
    payload_dict = _valid_payload()
    payload_dict["repository"]["name"] = "n" * 100
    payload_dict["repository"]["full_name"] = "o" * 99 + "/" + "n" * 100
    payload = PullRequestEventPayload.model_validate(payload_dict)
    assert len(payload.repository.name) == 100
    assert len(payload.repository.full_name) == 200


def test_sha_intermediate_length_rejected() -> None:
    """`sha` admits ONLY exactly 40 (SHA-1) or exactly 64 (SHA-256).
    Lengths 41-63 and 65+ are impossible per GitHub's spec; the prior
    range-bound `min_length=40 max_length=64` admitted them. The
    alternation pattern closes the gap."""
    for bad_len in (41, 42, 50, 63, 65, 80):
        payload_dict = _valid_payload()
        payload_dict["pull_request"]["head"]["sha"] = "a" * bad_len
        with pytest.raises(ValidationError):
            PullRequestEventPayload.model_validate(payload_dict)


def test_ref_leading_slash_rejected() -> None:
    """`ref` starting with `/` → ValidationError per git-check-ref-format.
    The char pattern alone admits this; the field_validator rejects."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["head"]["ref"] = "/main"
    with pytest.raises(ValidationError, match="cannot start with '/'"):
        PullRequestEventPayload.model_validate(payload_dict)


def test_ref_trailing_slash_rejected() -> None:
    """`ref` ending with `/` → ValidationError."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["head"]["ref"] = "feat/"
    with pytest.raises(ValidationError, match="cannot end with '/'"):
        PullRequestEventPayload.model_validate(payload_dict)


def test_ref_double_slash_rejected() -> None:
    """`ref` containing `//` → empty segment → ValidationError."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["head"]["ref"] = "feat//bar"
    with pytest.raises(ValidationError, match="empty segment"):
        PullRequestEventPayload.model_validate(payload_dict)


def test_ref_traversal_segment_rejected() -> None:
    """`ref` with `..` as a segment → ValidationError. Covers both
    `../head` (leading) and `foo/../bar` (interior). Char pattern
    admits both because `.` is in the class."""
    for bad in ["../head", "foo/../bar", ".."]:
        payload_dict = _valid_payload()
        payload_dict["pull_request"]["head"]["ref"] = bad
        with pytest.raises(ValidationError, match="'\\.\\.' traversal"):
            PullRequestEventPayload.model_validate(payload_dict)


def test_ref_embedded_double_dot_rejected() -> None:
    """`ref` segment containing `..` (not exactly `..`, not starting
    with `.`) → ValidationError. Covers `feature..x`, `a..b`, nested
    `release/foo..bar`. The `seg == ".."` check catches the bare form;
    `startswith(".")` catches `.hidden`; embedded `..` between non-dot
    chars slips both — git-check-ref-format rejects any `..` regardless
    of position."""
    for bad in ["feature..x", "a..b", "release/foo..bar"]:
        payload_dict = _valid_payload()
        payload_dict["pull_request"]["head"]["ref"] = bad
        with pytest.raises(ValidationError, match="contains '\\.\\.'"):
            PullRequestEventPayload.model_validate(payload_dict)


def test_ref_dot_prefix_segment_rejected() -> None:
    """`ref` segment starting with `.` → ValidationError per
    git-check-ref-format. Covers `.hidden`, `release/.tmp` — both pass
    the char pattern (`.` is allowed) AND the `..` segment check
    (these aren't exactly `..`) but git rejects any component
    starting with `.`. Without this rule dot-prefixed components
    cross the trust boundary."""
    for bad in [".hidden", "release/.tmp", "feat/.lock-name", "a/.b/c"]:
        payload_dict = _valid_payload()
        payload_dict["pull_request"]["head"]["ref"] = bad
        with pytest.raises(ValidationError, match="starting with '\\.'"):
            PullRequestEventPayload.model_validate(payload_dict)


def test_ref_lock_suffix_rejected() -> None:
    """`ref` segment ending in `.lock` → ValidationError per
    git-check-ref-format. Git reserves `.lock` suffixes for lock files."""
    for bad in ["feat.lock", "foo/bar.lock"]:
        payload_dict = _valid_payload()
        payload_dict["pull_request"]["head"]["ref"] = bad
        with pytest.raises(ValidationError, match="'.lock'"):
            PullRequestEventPayload.model_validate(payload_dict)


def test_ref_dot_suffix_rejected() -> None:
    """`ref` segment ending in `.` → ValidationError per
    git-check-ref-format."""
    for bad in ["foo.", "foo/bar."]:
        payload_dict = _valid_payload()
        payload_dict["pull_request"]["head"]["ref"] = bad
        with pytest.raises(ValidationError, match="ends with '.'"):
            PullRequestEventPayload.model_validate(payload_dict)


def test_name_with_whitespace_rejected() -> None:
    """`RepositoryRef.name` strict charset rejects whitespace. The
    prior `[^/]+` admitted any non-slash including spaces, tabs,
    newlines."""
    for bad in ["my repo", "my\trepo", "my\nrepo"]:
        payload_dict = _valid_payload()
        payload_dict["repository"]["name"] = bad
        with pytest.raises(ValidationError):
            PullRequestEventPayload.model_validate(payload_dict)


def test_name_with_shell_metachars_rejected() -> None:
    """`name` strict charset rejects shell-metacharacters that the
    prior `[^/]+` admitted. Same rationale as login: forged-payload
    indicator AND defense-in-depth on the URL-segment construction
    path."""
    for bad in ["repo;rm", "repo$x", "repo`x`", "repo|bad", "repo&&x"]:
        payload_dict = _valid_payload()
        payload_dict["repository"]["name"] = bad
        with pytest.raises(ValidationError):
            PullRequestEventPayload.model_validate(payload_dict)


def test_name_with_dot_dash_underscore_admitted() -> None:
    """Positive-boundary: real GitHub repo-name shapes (alphanumeric +
    `.` `_` `-`) admit cleanly. Pins the carve-out: the tightened
    pattern is restrictive enough to reject shell-meta but permissive
    enough for legitimate names."""
    for good in ["widgets", "my.repo", "my-repo", "my_repo", "Repo-2", "v1.0"]:
        payload_dict = _valid_payload()
        payload_dict["repository"]["name"] = good
        payload_dict["repository"]["full_name"] = f"acme/{good}"
        payload = PullRequestEventPayload.model_validate(payload_dict)
        assert payload.repository.name == good


def test_full_name_with_whitespace_or_shell_rejected() -> None:
    """`full_name` strict charset matches `name` strict charset on
    both halves of `<owner>/<name>`. Rejects whitespace + shell-meta
    that the prior `[^/]+/[^/]+` admitted."""
    for bad in ["acme/my repo", "acme/repo;rm", "ac me/widgets", "acme/widgets$"]:
        payload_dict = _valid_payload()
        payload_dict["repository"]["full_name"] = bad
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


def test_login_exactly_at_max_length_admitted() -> None:
    """Positive-boundary case: `max_length=39` is inclusive (Pydantic's
    semantics). A 39-char login admits cleanly. Pins the inclusive
    boundary so a strict-less-than regression flips this test."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["user"]["login"] = "a" * 39
    payload = PullRequestEventPayload.model_validate(payload_dict)
    assert len(payload.pull_request.user.login) == 39


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


def test_pr_title_exactly_at_max_length_admitted() -> None:
    """Positive-boundary case: `max_length=4096` is inclusive. A
    4096-char title admits cleanly. Pins the inclusive boundary so a
    strict-less-than regression flips this test."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["title"] = "x" * 4096
    payload = PullRequestEventPayload.model_validate(payload_dict)
    assert len(payload.pull_request.title) == 4096


def test_pr_body_exactly_at_max_length_admitted() -> None:
    """Positive-boundary case: `max_length=65536` is inclusive. A
    65536-char body admits cleanly."""
    payload_dict = _valid_payload()
    payload_dict["pull_request"]["body"] = "y" * 65536
    payload = PullRequestEventPayload.model_validate(payload_dict)
    assert payload.pull_request.body is not None
    assert len(payload.pull_request.body) == 65536


def test_models_are_frozen() -> None:
    """The parsed payload is immutable downstream of webhook parsing."""
    payload = PullRequestEventPayload.model_validate(_valid_payload())
    with pytest.raises(ValidationError):
        # `frozen=True` raises ValidationError on assignment when
        # `validate_assignment=False` is the implicit default. Pydantic v2
        # raises ValidationError for frozen models on field assignment.
        payload.action = "closed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Install-lifecycle event schemas (Arc B2, DECISIONS.md#065)
# ---------------------------------------------------------------------------


def _installation_obj() -> dict[str, Any]:
    return {
        "id": 12345,
        "account": {"id": 1, "login": "octocat", "type": "User"},
        "app_slug": "test-app",
        "repository_selection": "selected",
        "permissions": {"contents": "read", "pull_requests": "write"},
        "suspended_at": None,
    }


def test_installation_created_event_parses() -> None:
    """`installation.created` parses: account, app_slug, selection, and the granted repos —
    which use the MINIMAL repo shape (no `owner` sub-object)."""
    payload = InstallationEventPayload.model_validate(
        {
            "action": "created",
            "installation": _installation_obj(),
            "repositories": [{"id": 100, "full_name": "octocat/repo-a"}],
            "sender": {"login": "octocat", "id": 1},  # extra top-level -> ignored
        }
    )
    assert payload.action == "created"
    assert payload.installation.account.type == "User"
    assert payload.installation.repository_selection == "selected"
    assert len(payload.repositories) == 1
    assert payload.repositories[0].id == 100


def test_installation_repositories_added_event_parses() -> None:
    """`installation_repositories.added` parses the added/removed delta arrays + selection."""
    payload = InstallationRepositoriesEventPayload.model_validate(
        {
            "action": "added",
            "installation": _installation_obj(),
            "repository_selection": "selected",
            "repositories_added": [{"id": 200, "full_name": "octocat/repo-b"}],
            "repositories_removed": [],
        }
    )
    assert payload.action == "added"
    assert payload.repositories_added[0].full_name == "octocat/repo-b"
    assert payload.repositories_removed == ()


def test_installation_repository_ref_needs_no_owner() -> None:
    """The install-event repo shape has NO nested `owner` (unlike `RepositoryRef`)."""
    ref = InstallationRepositoryRef.model_validate({"id": 100, "full_name": "octocat/repo-a"})
    assert ref.id == 100


def test_installation_bad_account_login_rejected() -> None:
    """Input boundary: a slashed account login (path-injection shape) is rejected at parse."""
    bad = _installation_obj()
    bad["account"] = {"id": 1, "login": "octo/cat", "type": "User"}
    with pytest.raises(ValidationError):
        InstallationEventPayload.model_validate({"action": "created", "installation": bad})


def test_installation_bad_repo_full_name_rejected() -> None:
    """Input boundary: a repo `full_name` without the `owner/repo` shape is rejected."""
    with pytest.raises(ValidationError):
        InstallationRepositoryRef.model_validate({"id": 1, "full_name": "no-slash-here"})


def test_installation_action_is_open_string_not_literal() -> None:
    """`action` is `str` (not a closed Literal): an undocumented action parses cleanly and
    reaches the handler-side allowlist (which 2xx no-ops it) — no 400 that would cause retries."""
    payload = InstallationEventPayload.model_validate(
        {"action": "some_future_action", "installation": _installation_obj()}
    )
    assert payload.action == "some_future_action"
    assert payload.repositories == ()
