"""Startup settings for human-approval expiry (docs/spec.md §4.1.6 and §16).

V1 accepts only expire_only. The enum reserves auto_post, but construction
rejects it to preserve `hitl-gates-high-severity`.

build_graph injects these settings (`nodes-receive-deps-via-closure`). HITL
derives expiry from state.received_at plus timeout_minutes, so retries keep
the same request window (Q4 of the spec)."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HITLTimeoutAction(StrEnum):
    """Expiry modes; V1 construction accepts only EXPIRE_ONLY."""

    EXPIRE_ONLY = "expire_only"
    AUTO_POST = "auto_post"


class HITLConfig(BaseSettings):
    """Frozen settings from OUTRIDER_HITL_TIMEOUT_MINUTES (default 30) and
    OUTRIDER_HITL_TIMEOUT_ACTION (default expire_only).

    Timeout must be positive; auto_post is rejected before the graph is built.
    Tests can inject a short positive timeout without changing production defaults."""

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_HITL_",
        env_file=None,
        extra="forbid",
        frozen=True,
    )

    # Zero expires the request immediately; expiry tests use a positive timeout.
    timeout_minutes: int = Field(default=30, gt=0)
    timeout_action: HITLTimeoutAction = Field(default=HITLTimeoutAction.EXPIRE_ONLY)

    @model_validator(mode="after")
    def _enforce_v1_expire_only(self) -> Self:
        """Reject auto_post: V1 has no approved publication path for expiry.

        Fail at construction to preserve `hitl-gates-high-severity`."""
        if self.timeout_action != HITLTimeoutAction.EXPIRE_ONLY:
            raise ValueError(
                f"HITLConfig.timeout_action={self.timeout_action.value!r} is not "
                f"supported in V1. Only {HITLTimeoutAction.EXPIRE_ONLY.value!r} is "
                f"accepted; the `auto_post` mode is deferred to V1.5. To exit this "
                f"check, unset OUTRIDER_HITL_TIMEOUT_ACTION or set it to "
                f"{HITLTimeoutAction.EXPIRE_ONLY.value!r}."
            )
        return self


# The OUTRIDER_HITL_ prefix supplies the spec's timeout settings (§16).
__all__ = ["HITLConfig", "HITLTimeoutAction"]


# Silence unused-import warning for ConfigDict (we use SettingsConfigDict).
_ = ConfigDict
