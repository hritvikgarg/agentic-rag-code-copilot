"""Bounded exponential backoff for transient hosted-LLM failures (Milestone 6 evaluation runner).

Only three kinds of provider failure are worth a second attempt: an explicit rate/quota limit
(HTTP 429, ``LLMRateLimitError``), "service unavailable" (HTTP 503, ``LLMProviderError`` with
``status_code == 503``) and "gateway timeout" (HTTP 504, ``LLMTimeoutError`` with
``status_code == 504``) - each is the provider or its proxy saying "try again later", not "this
request is wrong". A client-side timeout (no HTTP response at all) and HTTP 408 are deliberately
NOT retried: the request would be identical and would most likely take just as long again.
Every other ``LLMError`` (bad configuration, bad credentials, an unknown model, a blocked or
malformed response, a 400) is a fixed condition that another identical attempt cannot fix, so it
is raised immediately. The
secret gate's ``RepositorySecretRiskError`` is not an ``LLMError`` at all and is never retried
here either: retrying would just repeat the same "unsafe to send" decision on the same content.

Why these defaults (chosen for the observed failure, not copied blindly): the first live 34
question run paced requests 1 second apart and still hit sustained HTTP 429 from Gemini (30/34
plain calls, 29/34 RAG calls). A single short wait rarely clears a *sustained* quota condition,
so the delay grows: 5s, 10s, 20s, 40s, capped at 60s. Five attempts (one try plus four retries)
bounds the worst case for one side of one question at 5+10+20+40 = 75s of extra waiting before
giving up and recording the error - long enough to ride out a short burst, short enough that a
run of 34 questions does not hang for hours on a single stuck question. There is no infinite
retry loop: after ``max_attempts`` the last error is raised and recorded, and a later ``--resume``
run can retry it again with a fresh budget.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from copilot.llm.errors import LLMError, LLMProviderError, LLMRateLimitError, LLMTimeoutError

logger = logging.getLogger(__name__)


DEFAULT_INITIAL_DELAY_SECONDS = 5.0
DEFAULT_MULTIPLIER = 2.0
DEFAULT_MAX_DELAY_SECONDS = 60.0
DEFAULT_MAX_ATTEMPTS = 5  # 1 initial try + 4 retries


@dataclass(frozen=True)
class RetryPolicy:
    """An explicit, testable backoff policy. Distinct from request *pacing* (``delay_seconds``
    in ``copilot.evaluation.comparison``), which paces well-behaved requests apart in time
    regardless of failure; this policy only governs what happens after a transient failure."""

    initial_delay_seconds: float = DEFAULT_INITIAL_DELAY_SECONDS
    multiplier: float = DEFAULT_MULTIPLIER
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    def __post_init__(self) -> None:
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must not be negative")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must not be negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1 (backoff must not shrink)")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1 (at least one attempt)")

    def delay_before_attempt(self, attempt: int) -> float:
        """The wait, in seconds, before retry number ``attempt`` (1 = the first retry)."""
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        delay = self.initial_delay_seconds * (self.multiplier ** (attempt - 1))
        return min(delay, self.max_delay_seconds)


DEFAULT_RETRY_POLICY = RetryPolicy()


def is_transient(exc: Exception) -> bool:
    """True only for the provider failures a second identical attempt can plausibly fix."""
    if isinstance(exc, LLMRateLimitError):
        return True
    if isinstance(exc, LLMTimeoutError):
        return exc.status_code == 504
    return isinstance(exc, LLMProviderError) and exc.status_code == 503


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    sleep: Callable[[float], None] = time.sleep,
    is_retryable: Callable[[Exception], bool] = is_transient,
    on_retry: Callable[[int, LLMError, float], None] | None = None,
) -> T:
    """Call ``fn`` up to ``policy.max_attempts`` times, retrying only ``is_retryable`` errors.

    A non-retryable ``LLMError`` propagates on the first attempt. A retryable one is retried with
    the policy's backoff, honouring the error's own ``retry_after_seconds`` (e.g. a Gemini
    Retry-After header) when it asks for longer than the computed delay - still capped at
    ``policy.max_delay_seconds``. After the final attempt the last error propagates unchanged so
    the caller can record it. Any non-``LLMError`` exception (including ``KeyboardInterrupt``,
    which is not even an ``Exception``) is never caught here and aborts immediately - a run
    interrupted mid-backoff leaves no partial state for this call to clean up.
    """
    attempt = 1
    while True:
        try:
            return fn()
        except LLMError as exc:
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            delay = policy.delay_before_attempt(attempt)
            retry_after = getattr(exc, "retry_after_seconds", None)
            if isinstance(retry_after, int | float) and retry_after > delay:
                delay = min(float(retry_after), policy.max_delay_seconds)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            if delay:
                sleep(delay)
            attempt += 1
