"""Shared model-call types, native Anthropic provider, logging, and pricing.

The OpenAI-compatible provider and host profiles are imported from their
own modules. Vendor SDK access stays within `llm/`.
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
