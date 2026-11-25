"""Provider-independent LLM interface: Fake, Gemini (hosted) and Ollama (local) clients.

Every call goes through ``GuardedLLMClient`` (the secret gate), including local ones.
See ``docs/llm.md``.
"""

from copilot.llm.base import LLMClient
from copilot.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMContextLimitError,
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
from copilot.llm.ollama import OllamaLLMClient
from copilot.llm.retry import RetryPolicy, call_with_retry, is_transient
from copilot.llm.runtime import RuntimeIdentity

__all__ = [
    "FakeLLMClient",
    "GeminiLLMClient",
    "GuardedLLMClient",
    "LLMAuthenticationError",
    "LLMClient",
    "LLMConfigurationError",
    "LLMConnectionError",
    "LLMContextLimitError",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseError",
    "LLMTimeoutError",
    "OllamaLLMClient",
    "RetryPolicy",
    "RuntimeIdentity",
    "call_with_retry",
    "create_llm_client",
    "is_transient",
]
