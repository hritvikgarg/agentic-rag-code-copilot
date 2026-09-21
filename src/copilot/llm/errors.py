"""Typed, secret-safe LLM errors.

Messages contain the provider name, an HTTP status code and a fixed explanation - never the
request text, the API key or the provider's raw error body (which can echo request content).
Provider exceptions are raised ``from None`` so a traceback cannot print them either.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every failure of the LLM layer."""


class LLMConfigurationError(LLMError):
    """The provider, model or API key is not configured (or the model does not exist)."""


class LLMAuthenticationError(LLMError):
    """The provider rejected the credentials (HTTP 401/403)."""


class LLMRateLimitError(LLMError):
    """The provider's rate or quota limit was hit (HTTP 429)."""


class LLMTimeoutError(LLMError):
    """The request did not finish within the configured timeout."""


class LLMConnectionError(LLMError):
    """The provider could not be reached (network/DNS/TLS)."""


class LLMProviderError(LLMError):
    """The provider failed or rejected the request for another reason."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMResponseError(LLMError):
    """The provider answered, but with no usable text (empty, blocked or malformed)."""
