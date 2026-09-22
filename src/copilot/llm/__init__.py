"""Provider-independent LLM interface (Milestone 6): Fake and Gemini clients plus the secret gate.

Only the Gemini provider is implemented. Every hosted call goes through ``GuardedLLMClient``.
See ``docs/llm.md``.
"""

from copilot.llm.base import LLMClient
from copilot.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from copilot.llm.factory import create_llm_client
from copilot.llm.fake import FakeLLMClient
from copilot.llm.gemini import GeminiLLMClient
from copilot.llm.guarded import GuardedLLMClient
from copilot.llm.models import LLMRequest, LLMResponse
from copilot.llm.retry import RetryPolicy, call_with_retry, is_transient

__all__ = [
    "FakeLLMClient",
    "GeminiLLMClient",
    "GuardedLLMClient",
    "LLMAuthenticationError",
    "LLMClient",
    "LLMConfigurationError",
    "LLMConnectionError",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseError",
    "LLMTimeoutError",
    "RetryPolicy",
    "call_with_retry",
    "create_llm_client",
    "is_transient",
]
