"""LLM provider wrapper — public surface.

Commit-1 export set per `specs/2026-05-05-llm-provider-wrapper.md`
§Commit boundary. `AnthropicProvider` is NOT exported here yet
(lands in commit 2 with `anthropic_provider.py`); commit 2 extends
this list.
"""

from outrider.llm.base import (
    INCLUDE_TEXT_OPT_IN,
    LLMAuthError,
    LLMExchangePersister,
    LLMInvalidRequestError,
    LLMInvalidResponseError,
    LLMMessage,
    LLMMissingAPIKeyError,
    LLMPersisterError,
    LLMPersisterNotWiredError,
    LLMPricingMissingError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMUnexpectedContentBlocksError,
    LLMUnknownError,
    LLMUpstreamError,
    RetryLayer,
)
from outrider.llm.config import ModelConfig
from outrider.llm.logging import (
    LLM_CONTENT_BEARING_TYPES,
    RejectLLMContentFilter,
    register_filter_on_all_handlers,
)

__all__ = [
    # Sentinel
    "INCLUDE_TEXT_OPT_IN",
    # Logging filter + constant
    "LLM_CONTENT_BEARING_TYPES",
    # Exception hierarchy
    "LLMAuthError",
    "LLMInvalidRequestError",
    "LLMInvalidResponseError",
    "LLMMissingAPIKeyError",
    "LLMPersisterError",
    "LLMPersisterNotWiredError",
    "LLMPricingMissingError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMUnexpectedContentBlocksError",
    "LLMUnknownError",
    "LLMUpstreamError",
    # Protocols
    "LLMExchangePersister",
    "LLMProvider",
    # Schemas
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    # Config
    "ModelConfig",
    # Filter
    "RejectLLMContentFilter",
    "RetryLayer",
    "register_filter_on_all_handlers",
]
