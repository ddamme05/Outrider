"""RetentionSettings unit tests — TTL constraints + env-var override.

Pins the `gt=timedelta(0)` constraint that prevents the "operator sets
0/negative TTL by accident, retention sweep deletes every row on next
tick" failure mode (per `specs/2026-05-16-audit-persister.md` H1).
"""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from outrider.audit.config import RetentionSettings

# ---------------------------------------------------------------------------
# Defaults + explicit construction.
# ---------------------------------------------------------------------------


def test_default_ttl_is_90_days() -> None:
    """Default TTL matches DECISIONS#016's architectural anchor (90 days)."""
    settings = RetentionSettings()
    assert settings.llm_content_retention_ttl == timedelta(days=90)


def test_findings_ttl_default_is_90_days() -> None:
    """findings_retention_ttl defaults to 90 days (equal to llm + review defaults
    so the llm <= findings <= review ordering holds at the defaults)."""
    settings = RetentionSettings()
    assert settings.findings_retention_ttl == timedelta(days=90)


def test_findings_ttl_explicit_override() -> None:
    """findings_retention_ttl is operator-overridable via explicit kwarg.

    180d findings with a 180d review satisfies the ordering; the value the
    replay docstrings historically narrated is reachable when the operator
    also raises the review TTL.
    """
    settings = RetentionSettings(
        findings_retention_ttl=timedelta(days=180),
        review_retention_ttl=timedelta(days=180),
    )
    assert settings.findings_retention_ttl == timedelta(days=180)


def test_explicit_kwarg_overrides_default() -> None:
    """Explicit kwarg construction works and overrides the default."""
    settings = RetentionSettings(llm_content_retention_ttl=timedelta(days=7))
    assert settings.llm_content_retention_ttl == timedelta(days=7)


# ---------------------------------------------------------------------------
# gt=timedelta(0) constraint — rejects zero and negative.
# ---------------------------------------------------------------------------


def test_zero_ttl_raises_validation_error() -> None:
    """Zero TTL is rejected at construction (would silently purge everything
    on the next retention sweep tick if accepted)."""
    with pytest.raises(ValidationError):
        RetentionSettings(llm_content_retention_ttl=timedelta(0))


def test_negative_ttl_raises_validation_error() -> None:
    """Negative TTL is rejected at construction (rows would be expired on
    insert; sweep would delete them before any reader could see them)."""
    with pytest.raises(ValidationError):
        RetentionSettings(llm_content_retention_ttl=timedelta(seconds=-1))


def test_one_second_ttl_is_accepted_by_constraint() -> None:
    """`gt=timedelta(0)` is strictly greater — 1 second satisfies.

    Note: the spec deliberately does NOT add a sub-minute floor; operators
    are trusted to configure sensibly. The constraint is "strictly positive",
    not "production-realistic". A future hardening could add the floor; not
    in scope here per Codex's call.
    """
    settings = RetentionSettings(llm_content_retention_ttl=timedelta(seconds=1))
    assert settings.llm_content_retention_ttl == timedelta(seconds=1)


# ---------------------------------------------------------------------------
# Retention ordering: llm_content <= findings <= review.
# ---------------------------------------------------------------------------


def test_ordering_holds_at_defaults() -> None:
    """All three default to 90d, so llm <= findings <= review holds."""
    settings = RetentionSettings()
    assert (
        settings.llm_content_retention_ttl
        <= settings.findings_retention_ttl
        <= settings.review_retention_ttl
    )


def test_findings_ttl_exceeding_review_raises() -> None:
    """findings > review violates the ordering: findings.review_id is
    ON DELETE CASCADE, so a longer findings TTL is unreachable AND would
    manufacture the review-purged/content-survives corruption state."""
    with pytest.raises(ValidationError):
        RetentionSettings(
            findings_retention_ttl=timedelta(days=120),
            review_retention_ttl=timedelta(days=90),
        )


def test_llm_content_exceeding_findings_raises() -> None:
    """llm_content > findings violates the ordering: the most-sensitive
    content (prompt/completion text) must carry the shortest TTL."""
    with pytest.raises(ValidationError):
        RetentionSettings(
            llm_content_retention_ttl=timedelta(days=120),
            findings_retention_ttl=timedelta(days=90),
        )


def test_strict_ascending_ordering_is_accepted() -> None:
    """A genuine llm < findings < review ladder constructs cleanly."""
    settings = RetentionSettings(
        llm_content_retention_ttl=timedelta(days=30),
        findings_retention_ttl=timedelta(days=90),
        review_retention_ttl=timedelta(days=180),
    )
    assert settings.llm_content_retention_ttl == timedelta(days=30)
    assert settings.findings_retention_ttl == timedelta(days=90)
    assert settings.review_retention_ttl == timedelta(days=180)


# ---------------------------------------------------------------------------
# Env-var override (per DECISIONS#012 operator-overridable contract).
# ---------------------------------------------------------------------------


def test_env_var_iso8601_duration_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL=P7D` parses as 7 days."""
    monkeypatch.setenv("OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL", "P7D")
    settings = RetentionSettings()
    assert settings.llm_content_retention_ttl == timedelta(days=7)


def test_env_var_hour_form_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """ISO-8601 hour-precision form: `PT24H` parses as 1 day.

    Documents the env-var format operators should use. Bare integers
    (e.g., "604800" for seconds) are NOT accepted by pydantic-settings
    2.13.1's timedelta parser — operators must use ISO-8601 duration syntax.
    """
    monkeypatch.setenv("OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL", "PT24H")
    settings = RetentionSettings()
    assert settings.llm_content_retention_ttl == timedelta(days=1)


def test_env_var_bare_integer_seconds_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pydantic-settings 2.13.1 does NOT parse bare integer env var as
    seconds for `timedelta` fields — only ISO-8601 duration strings
    (`P7D`, `PT24H`) are accepted. Test documents the operator-facing
    format constraint so a future pydantic-settings upgrade that DOES
    accept bare ints surfaces as a test failure (welcome relaxation),
    not silent interpretation drift.
    """
    monkeypatch.setenv("OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL", "604800")
    with pytest.raises(ValidationError):
        RetentionSettings()


def test_env_var_zero_is_rejected_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator setting `OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL=0` fails
    loud at startup, not silently-purges-everything at next sweep tick."""
    monkeypatch.setenv("OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL", "0")
    with pytest.raises(ValidationError):
        RetentionSettings()


def test_env_var_negative_iso_duration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative ISO-8601 duration (`-P1D`) is rejected."""
    monkeypatch.setenv("OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL", "-P1D")
    with pytest.raises(ValidationError):
        RetentionSettings()


# ---------------------------------------------------------------------------
# Pydantic model_config: extra=forbid + frozen=True.
# ---------------------------------------------------------------------------


def test_unknown_field_kwarg_raises() -> None:
    """`extra="forbid"` catches typos at construction."""
    with pytest.raises(ValidationError):
        RetentionSettings(  # type: ignore[call-arg]
            llm_content_retention_ttl=timedelta(days=7),
            unknown_field="surprise",
        )


def test_frozen_means_attribute_assignment_raises() -> None:
    """`frozen=True` blocks post-construction mutation. The pattern mirrors
    `ModelConfig`: re-construct with new kwargs rather than mutate."""
    settings = RetentionSettings(llm_content_retention_ttl=timedelta(days=7))
    with pytest.raises(ValidationError):
        settings.llm_content_retention_ttl = timedelta(days=14)  # type: ignore[misc]


def test_env_prefix_is_subsystem_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var without the `OUTRIDER_AUDIT_` prefix is ignored.

    Regression test for the spec drift fix (M1): the prefix is
    subsystem-scoped (`OUTRIDER_AUDIT_`), matching the `ModelConfig`
    precedent (`OUTRIDER_MODEL_`). A bare `OUTRIDER_LLM_CONTENT_RETENTION_TTL`
    (the wrong-prefix form) must NOT be read.
    """
    monkeypatch.setenv("OUTRIDER_LLM_CONTENT_RETENTION_TTL", "P1D")  # wrong prefix
    monkeypatch.delenv("OUTRIDER_AUDIT_LLM_CONTENT_RETENTION_TTL", raising=False)
    settings = RetentionSettings()
    # Default applies; the misprefixed env var was ignored.
    assert settings.llm_content_retention_ttl == timedelta(days=90)
