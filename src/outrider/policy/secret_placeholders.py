"""Known public placeholder values that cannot serve as deployment secrets.

Consumers own normalization, length checks and failure behavior. Slack token
settings also reject their token-specific example value.
"""

from typing import Final

SECRET_PLACEHOLDERS: Final[frozenset[str]] = frozenset(
    {
        "replace-me",
        "replace-me-with-a-long-random-secret",
        "change-me",
        "changeme",
        "secret",
        "password",
        "your-secret-here",
    }
)
