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
    """The provider's rate or quota limit was hit (HTTP 429).

    ``retry_after_seconds`` is the provider's own requested wait, read from a ``Retry-After``
    response header when the SDK exposes one (``None`` otherwise - Gemini does not always send
    it). It is a plain number of seconds, never request content, so it is always safe to log.
    """

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMTimeoutError(LLMError):
    """The request did not finish in time.

    ``status_code`` is set only when the *provider* said so over HTTP (408 request timeout, 504
    gateway timeout); it is ``None`` for a client-side timeout where no response arrived at all.
    The distinction matters for retries: see ``copilot.llm.retry.is_transient``.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMConnectionError(LLMError):
    """The provider could not be reached (network/DNS/TLS)."""


class LLMProviderError(LLMError):
    """The provider failed or rejected the request for another reason.

    ``status_code`` is the HTTP status when known. Most causes here are not transient (a bad
    request, an unexpected SDK failure), but HTTP 503 ("service unavailable") is - see
    ``copilot.llm.retry.is_transient``. ``retry_after_seconds`` mirrors ``LLMRateLimitError``'s.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class LLMResponseError(LLMError):
    """The provider answered, but with no usable text (empty, blocked or malformed)."""


class LLMContextLimitError(LLMError):
    """The request or response cannot be trusted to fit the model's context window.

    Raised *before* sending when the conservative input estimate exceeds the budget, and *after*
    the answer when the provider's own metadata says the output may have been cut short or the
    context may have been truncated or shifted. The message contains token counts and a fixed
    explanation only - never request text. A failure of this kind is never silently repaired
    (no prompt shortening, no discarded evidence): it is reported and the item is excluded from
    the valid-judgement counts.
    """
