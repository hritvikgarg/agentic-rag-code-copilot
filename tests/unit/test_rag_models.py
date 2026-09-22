"""GenerationInfo: safe, diagnostic-only metadata carried from an LLMResponse into an answer."""

from copilot.llm import LLMResponse
from copilot.rag.models import GenerationInfo


def test_from_response_copies_safe_fields_including_reasoning_tokens():
    response = LLMResponse(
        text="hi",
        provider="gemini",
        model="model-x",
        finish_reason="MAX_TOKENS",
        prompt_tokens=3201,
        completion_tokens=41,
        total_tokens=4221,
        reasoning_tokens=979,
        latency_seconds=8.648,
    )
    info = GenerationInfo.from_response(response)
    assert info.provider == "gemini" and info.model == "model-x"
    assert info.finish_reason == "MAX_TOKENS"
    assert (info.prompt_tokens, info.completion_tokens, info.total_tokens) == (3201, 41, 4221)
    assert info.reasoning_tokens == 979
    assert info.latency_seconds == 8.648


def test_from_response_tolerates_no_reasoning_tokens():
    response = LLMResponse(text="hi", provider="fake", model="m")
    assert GenerationInfo.from_response(response).reasoning_tokens is None


def test_generation_info_never_carries_request_text():
    response = LLMResponse(text="private-repo-secret-marker", provider="fake", model="m")
    info = GenerationInfo.from_response(response)
    assert "private-repo-secret-marker" not in repr(
        info
    ) and "private-repo-secret-marker" not in str(info.model_dump())
