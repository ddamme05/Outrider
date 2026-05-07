"""LLM provider wrapper — public surface.

Full export set per `specs/2026-05-05-llm-provider-wrapper.md`
§Commit boundary (commit 2 extension: adds `AnthropicProvider` +
pricing surfaces).
"""

from outrider.llm.anthropic_provider import AnthropicProvider
from outrider.llm.base import (
    INCLUDE_TEXT_OPT_IN,
    LLMAuthError,
    LLMConflictError,
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
from outrider.llm.pricing import (
    PRICING_VERSION,
    RATE_TABLE,
    ModelPricing,
    compute_cost_usd,
)

__all__ = [
    # Sentinel
    "INCLUDE_TEXT_OPT_IN",
    # Logging filter + constant
    "LLM_CONTENT_BEARING_TYPES",
    # Exception hierarchy
    "LLMAuthError",
    "LLMConflictError",
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
    "PRICING_VERSION",
    "RATE_TABLE",
    "AnthropicProvider",
    "ModelPricing",
    "RejectLLMContentFilter",
    "RetryLayer",
    "compute_cost_usd",
    "register_filter_on_all_handlers",
]
