"""Retention policy shared by installation webhooks and reconciliation."""

from datetime import timedelta
from typing import Final

# Preserve the recovery window after an uninstall or a missed deletion webhook.
# Operator configuration remains tracked separately under FUP-226.
INSTALLATION_TOMBSTONE_GRACE: Final[timedelta] = timedelta(days=30)
