"""Bounded exponential backoff: which errors are retried, the delay schedule, no real waiting."""

import pytest

from copilot.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from copilot.llm.retry import RetryPolicy, call_with_retry, is_transient


def test_only_rate_limit_503_and_504_are_transient():
    assert is_transient(LLMRateLimitError("429"))
    assert is_transient(LLMProviderError("503", status_code=503))
    assert not is_transient(LLMProviderError("500", status_code=500))
    assert not is_transient(LLMProviderError("400", status_code=400))
    assert not is_transient(LLMConfigurationError("no model"))
    assert not is_transient(LLMResponseError("blocked"))
    assert is_transient(LLMTimeoutError("gateway timeout", status_code=504))


def test_everything_else_is_not_transient():
    # a client-side timeout (no HTTP response) and HTTP 408 are deliberately not retried
    assert not is_transient(LLMTimeoutError("client timeout"))
    assert not is_transient(LLMTimeoutError("request timeout", status_code=408))
    assert not is_transient(LLMAuthenticationError("401"))
    assert not is_transient(LLMConnectionError("dns"))
    assert not is_transient(LLMProviderError("unknown", status_code=None))


def test_a_504_is_retried_with_backoff_then_succeeds_without_real_sleep():
    calls: list[int] = []
    sleeps: list[float] = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise LLMTimeoutError("Gemini request timed out (HTTP 504)", status_code=504)
        return "ok"

    result = call_with_retry(
        flaky, policy=RetryPolicy(initial_delay_seconds=1.0, multiplier=2.0), sleep=sleeps.append
    )
    assert result == "ok" and len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_a_persistent_504_is_bounded_by_max_attempts():
    calls: list[int] = []

    def always():
        calls.append(1)
        raise LLMTimeoutError("HTTP 504", status_code=504)

    with pytest.raises(LLMTimeoutError):
        call_with_retry(always, policy=RetryPolicy(max_attempts=3), sleep=lambda _s: None)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "error",
    [
        LLMAuthenticationError("HTTP 401"),
        LLMConfigurationError("no model"),
        LLMProviderError("bad request", status_code=400),
        LLMResponseError("blocked"),
        LLMTimeoutError("client timeout"),
        LLMTimeoutError("HTTP 408", status_code=408),
        ValueError("deterministic validation failure"),  # not an LLMError at all
    ],
)
def test_non_retryable_failures_are_raised_on_the_first_attempt(error):
    calls: list[int] = []
    sleeps: list[float] = []

    def failing():
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        call_with_retry(failing, sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []  # one attempt, no waiting


def test_delay_schedule_grows_and_is_capped():
    policy = RetryPolicy(initial_delay_seconds=5.0, multiplier=2.0, max_delay_seconds=60.0)
    assert [policy.delay_before_attempt(a) for a in (1, 2, 3, 4, 5)] == [
        5.0,
        10.0,
        20.0,
        40.0,
        60.0,
    ]


def test_policy_rejects_nonsensical_values():
    with pytest.raises(ValueError, match="negative"):
        RetryPolicy(initial_delay_seconds=-1)
    with pytest.raises(ValueError, match="negative"):
        RetryPolicy(max_delay_seconds=-1)
    with pytest.raises(ValueError, match=">= 1"):
        RetryPolicy(multiplier=0.5)
    with pytest.raises(ValueError, match=">= 1"):
        RetryPolicy(max_attempts=0)


def test_a_transient_error_is_retried_then_succeeds_with_no_real_sleep():
    calls = []
    sleeps: list[float] = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise LLMRateLimitError("HTTP 429")
        return "ok"

    result = call_with_retry(
        flaky, policy=RetryPolicy(initial_delay_seconds=1.0, multiplier=2.0), sleep=sleeps.append
    )
    assert result == "ok"
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]  # two waits before the third (successful) attempt


def test_a_transient_503_is_retried_then_succeeds():
    calls = []
    sleeps: list[float] = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise LLMProviderError("HTTP 503", status_code=503)
        return "ok"

    result = call_with_retry(flaky, sleep=sleeps.append)
    assert result == "ok" and len(calls) == 2 and sleeps == [5.0]


def test_a_non_transient_error_is_never_retried():
    calls = []
    sleeps: list[float] = []

    def always_fails():
        calls.append(1)
        raise LLMConfigurationError("no GEMINI_API_KEY")

    with pytest.raises(LLMConfigurationError):
        call_with_retry(always_fails, sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []


def test_retry_exhaustion_raises_the_last_error_without_extra_attempts():
    calls = []
    sleeps: list[float] = []

    def always_rate_limited():
        calls.append(1)
        raise LLMRateLimitError("HTTP 429")

    policy = RetryPolicy(initial_delay_seconds=1.0, multiplier=1.0, max_attempts=3)
    with pytest.raises(LLMRateLimitError):
        call_with_retry(always_rate_limited, policy=policy, sleep=sleeps.append)
    assert len(calls) == 3  # 1 initial try + 2 retries, then give up
    assert sleeps == [1.0, 1.0]


def test_max_attempts_of_one_means_no_retries():
    calls = []
    sleeps: list[float] = []

    def always_fails():
        calls.append(1)
        raise LLMRateLimitError("429")

    with pytest.raises(LLMRateLimitError):
        call_with_retry(always_fails, policy=RetryPolicy(max_attempts=1), sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []


def test_on_retry_callback_receives_attempt_error_and_delay():
    seen: list[tuple[int, Exception, float]] = []
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise LLMRateLimitError("HTTP 429")
        return "ok"

    call_with_retry(
        flaky,
        policy=RetryPolicy(initial_delay_seconds=1.0, multiplier=2.0),
        sleep=lambda _: None,
        on_retry=lambda attempt, exc, delay: seen.append((attempt, exc, delay)),
    )
    assert [(a, type(e).__name__, d) for a, e, d in seen] == [
        (1, "LLMRateLimitError", 1.0),
        (2, "LLMRateLimitError", 2.0),
    ]


def test_a_generous_retry_after_header_extends_the_wait_but_stays_capped():
    calls = []
    sleeps: list[float] = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise LLMRateLimitError("HTTP 429", retry_after_seconds=45.0)
        return "ok"

    call_with_retry(
        flaky,
        policy=RetryPolicy(initial_delay_seconds=1.0, max_delay_seconds=60.0),
        sleep=sleeps.append,
    )
    assert sleeps == [45.0]  # the provider's own request beats the small computed backoff

    calls.clear()
    sleeps.clear()

    def flaky_over_cap():
        calls.append(1)
        if len(calls) < 2:
            raise LLMRateLimitError("HTTP 429", retry_after_seconds=500.0)
        return "ok"

    call_with_retry(
        flaky_over_cap,
        policy=RetryPolicy(initial_delay_seconds=1.0, max_delay_seconds=60.0),
        sleep=sleeps.append,
    )
    assert sleeps == [60.0]  # never waits longer than the policy's own cap
