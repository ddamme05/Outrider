"""Configure the saved review summary at startup."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class SummaryConfig(BaseSettings):
    """Use the finding count and triage risk when OUTRIDER_BRIEF_SUMMARY is enabled."""

    model_config = SettingsConfigDict(env_prefix="OUTRIDER_", extra="forbid", frozen=True)

    brief_summary: bool = False
