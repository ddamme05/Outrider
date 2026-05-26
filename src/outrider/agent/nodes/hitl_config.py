"""HITL node configuration: timeout window + timeout-action mode.

Canonical configuration setting per `docs/spec.md` §4.1.6
("Timeout handling") + §16 (demo overrides at line 1421). V1 ships
the `HITL_TIMEOUT_ACTION` knob with `expire_only` as the only
ACCEPTED value: the enum carries `auto_post` for forward-compatibility
with V1.5 but the startup validator raises `ValueError` if the env
var is set to `auto_post`. This preserves the `hitl-gates-high-severity`
absolute guarantee while shipping the canonical knob.

Closure-injected at `build_graph(...)` time per the
`nodes-receive-deps-via-closure` invariant; the HITL node reads
`hitl_config.timeout_minutes` for the deterministic
`expires_at = state.received_at + timedelta(minutes=...)` derivation
that's load-bearing for replay equivalence (per Q4 of the spec).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HITLTimeoutAction(StrEnum):
    """Action the sweep takes when an `HITLRequest` expires.

    Per `docs/spec.md` §4.1.6 "Timeout handling": the V1 implementation
    accepts ONLY `EXPIRE_ONLY` at startup; `AUTO_POST` exists in the
    enum for forward-compatibility but the startup validator raises
    if it's configured (V1 lifts the rejection when the `auto_post`
    impl lands).
    """

    EXPIRE_ONLY = "expire_only"
    AUTO_POST = "auto_post"


class HITLConfig(BaseSettings):
    """Closure-injected HITL config (timeout window + action mode).

    Reads `OUTRIDER_HITL_TIMEOUT_MINUTES` (default 30) and
    `OUTRIDER_HITL_TIMEOUT_ACTION` (default `expire_only`) from the
    environment per the demo `.env` override pattern at `docs/spec.md`
    §16. Startup validator (`_enforce_v1_expire_only`) raises
    `ValueError` if `timeout_action == AUTO_POST` — V1's
    `hitl-gates-high-severity` guarantee depends on no
    auto-post-on-expiry path existing.

    Tests construct `HITLConfig(timeout_minutes=0, ...)` directly and
    inject through `build_graph(...)` to exercise expiry paths without
    waiting 30 minutes.
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_HITL_",
        env_file=None,
        extra="forbid",
        frozen=True,
    )

    timeout_minutes: int = Field(default=30, ge=0)
    timeout_action: HITLTimeoutAction = Field(default=HITLTimeoutAction.EXPIRE_ONLY)

    @model_validator(mode="after")
    def _enforce_v1_expire_only(self) -> Self:
        """V1 supports only `expire_only`; `auto_post` raises at startup.

        The enum admits `AUTO_POST` for forward-compatibility but the
        V1 implementation has no auto-post-on-expiry path. Raising at
        startup is fail-loud per the `hitl-gates-high-severity`
        absolute-guarantee story: a misconfigured env var should
        prevent process boot, not silently admit unapproved
        CRITICAL/HIGH findings to GitHub on expiry.
        """
        if self.timeout_action != HITLTimeoutAction.EXPIRE_ONLY:
            raise ValueError(
                f"HITLConfig.timeout_action={self.timeout_action.value!r} is not "
                f"supported in V1. Only {HITLTimeoutAction.EXPIRE_ONLY.value!r} is "
                f"accepted; the `auto_post` mode is deferred to V1.5. To exit this "
                f"check, unset OUTRIDER_HITL_TIMEOUT_ACTION or set it to "
                f"{HITLTimeoutAction.EXPIRE_ONLY.value!r}."
            )
        return self


# `HITL_TIMEOUT_MINUTES` is the canonical env-var name per `docs/spec.md`
# §16 line 1421; we keep the canonical naming on `OUTRIDER_HITL_TIMEOUT_MINUTES`
# (Pydantic prefix `OUTRIDER_HITL_` + field `timeout_minutes`). The
# default of 30 minutes matches the canonical record.
__all__ = ["HITLConfig", "HITLTimeoutAction"]


# Silence unused-import warning for ConfigDict (we use SettingsConfigDict).
_ = ConfigDict
