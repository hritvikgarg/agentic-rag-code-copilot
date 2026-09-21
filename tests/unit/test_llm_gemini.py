"""The Gemini client with an injected fake SDK client: mapping, redaction, malformed replies."""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from pydantic import SecretStr

from copilot.llm import (
    GeminiLLMClient,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseError,
    LLMTimeoutError,
)
from copilot.llm.gemini import map_api_error

API_KEY = "test-key-not-real-0123456789"
RAW_BODY = "RAW-PROVIDER-BODY-ECHOING-repository-code"


def request() -> LLMRequest:
    return LLMRequest(system_instruction="sys", user_prompt="user", prompt_version="t-v1")


class FakeSdk:
    def __init__(self, reply=None, error: Exception | None = None):
        self.calls: list[dict] = []
        self._reply, self._error = reply, error
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._reply


def reply(text="hello", finish="STOP", usage=(3, 4, 7), block=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=usage[0], candidates_token_count=usage[1], total_token_count=usage[2]
        ),
        prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name=block))
        if block
        else None,
    )


def client(sdk) -> GeminiLLMClient:
    return GeminiLLMClient(api_key=SecretStr(API_KEY), model="model-x", sdk_client=sdk)


def api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"message": RAW_BODY, "status": "X"}})


def test_a_normal_reply_becomes_a_typed_response_with_usage():
    sdk = FakeSdk(reply=reply())
    response = client(sdk).generate(request())
    assert (response.text, response.provider, response.model) == ("hello", "gemini", "model-x")
    assert response.finish_reason == "STOP"
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (3, 4, 7)
    call = sdk.calls[0]
    assert call["model"] == "model-x" and call["contents"] == "user"
    assert call["config"].system_instruction == "sys"
    assert call["config"].temperature == 0.0 and call["config"].max_output_tokens == 1024


def test_the_key_never_appears_in_repr_or_str():
    c = client(FakeSdk(reply=reply()))
    assert API_KEY not in repr(c) and API_KEY not in str(c)
    assert "model-x" in repr(c)


def test_missing_model_or_key_is_a_configuration_error():
    with pytest.raises(LLMConfigurationError):
        GeminiLLMClient(api_key=SecretStr(API_KEY), model="")
    with pytest.raises(LLMConfigurationError):
        GeminiLLMClient(api_key=None, model="m")
    with pytest.raises(LLMConfigurationError):
        GeminiLLMClient(api_key=SecretStr(""), model="m")


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (401, LLMAuthenticationError),
        (403, LLMAuthenticationError),
        (404, LLMConfigurationError),
        (429, LLMRateLimitError),
        (504, LLMTimeoutError),
        (400, LLMProviderError),
        (500, LLMProviderError),
        (None, LLMProviderError),
    ],
)
def test_http_status_maps_to_a_typed_error(code, expected):
    assert type(map_api_error(code)) is expected


@pytest.mark.parametrize("code", [400, 401, 404, 429, 500])
def test_provider_errors_are_mapped_and_never_leak_the_raw_body_or_chain(code):
    with pytest.raises(Exception) as info:
        client(FakeSdk(error=api_error(code))).generate(request())
    assert RAW_BODY not in str(info.value) and API_KEY not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__
    assert type(info.value).__module__.startswith("copilot.llm")


def test_timeouts_and_network_failures_are_typed():
    with pytest.raises(LLMTimeoutError):
        client(FakeSdk(error=httpx.ReadTimeout("slow " + RAW_BODY))).generate(request())
    with pytest.raises(LLMConnectionError) as info:
        client(FakeSdk(error=httpx.ConnectError(RAW_BODY))).generate(request())
    assert RAW_BODY not in str(info.value)


def test_unexpected_sdk_exceptions_are_wrapped_without_their_message():
    with pytest.raises(LLMProviderError) as info:
        client(FakeSdk(error=RuntimeError(RAW_BODY + API_KEY))).generate(request())
    assert RAW_BODY not in str(info.value) and API_KEY not in str(info.value)
    assert "RuntimeError" in str(info.value)


@pytest.mark.parametrize("text", [None, "", "   \n"])
def test_empty_or_missing_text_is_a_response_error(text):
    with pytest.raises(LLMResponseError, match="no text"):
        client(FakeSdk(reply=reply(text=text, finish="MAX_TOKENS"))).generate(request())


def test_a_max_tokens_empty_reply_explains_the_setting():
    with pytest.raises(LLMResponseError, match="MAX_TOKENS.*MAX_OUTPUT_TOKENS"):
        client(FakeSdk(reply=reply(text="", finish="MAX_TOKENS"))).generate(request())


def test_a_blocked_prompt_is_a_response_error():
    with pytest.raises(LLMResponseError, match="blocked"):
        client(FakeSdk(reply=reply(text=None, block="SAFETY"))).generate(request())


def test_missing_usage_metadata_is_tolerated():
    raw = SimpleNamespace(text="hi", candidates=[], usage_metadata=None, prompt_feedback=None)
    response = client(FakeSdk(reply=raw)).generate(request())
    assert (
        response.text == "hi" and response.total_tokens is None and response.finish_reason is None
    )


def test_the_timeout_is_configured_in_milliseconds():
    c = GeminiLLMClient(api_key=SecretStr(API_KEY), model="m", timeout_seconds=12.5, sdk_client=1)
    assert c._timeout_ms == 12500  # noqa: SLF001
