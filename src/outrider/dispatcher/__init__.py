"""Queue a review after the webhook has accepted its request.

Dispatchers receive seed review data and run the workflow within a shared
concurrency limit. PRContext is built before scheduling, so a task needs no
open request session.

See DECISIONS.md#020, #045 and FUP-164.
"""

from outrider.dispatcher.background_tasks import BackgroundTasksDispatcher
from outrider.dispatcher.base import ReviewDispatcher
from outrider.dispatcher.concurrency import concurrency_limited
from outrider.dispatcher.config import DispatchConfig

__all__ = [
    "BackgroundTasksDispatcher",
    "DispatchConfig",
    "ReviewDispatcher",
    "concurrency_limited",
]
