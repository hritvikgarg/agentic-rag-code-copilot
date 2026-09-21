"""LLM request/response models, the fake client, the guarded wrapper and the factory."""

import pytest
from pydantic import ValidationError

from copilot.config import Settings
from copilot.llm import (
    FakeLLMClient,
    GuardedLLMClient,
    LLMClient,
    LLMConfigurationError,
    LLMRequest,
    create_llm_client,
)
from copilot.security import RepositorySecretRiskError
from tests.security import secret_helpers as h


def request(user: str = "What does ingest do?", system: str = "Be brief.") -> LLMRequest:
    return LLMRequest(system_instruction=system, user_prompt=user, prompt_version="test-v1")


def test_request_exposes_exactly_the_outbound_text_and_scan_targets():
    req = request(user="USER-TEXT", system="SYSTEM-TEXT")
    assert req.outbound_text() == "SYSTEM-TEXT\n\nUSER-TEXT"
    targets = req.scan_targets()
    assert [t.text for t in targets] == ["SYSTEM-TEXT", "USER-TEXT"]
    assert all(not t.path.startswith("/") for t in targets)


def test_request_repr_does_not_contain_the_prompt_text():
    text = repr(request(user="very-private-repository-code", system="also-private"))
    assert "very-private-repository-code" not in text and "also-private" not in text
    assert "test-v1" in text


@pytest.mark.parametrize("field", ["system_instruction", "user_prompt"])
def test_empty_prompt_parts_are_rejected(field):
    values = {"system_instruction": "s", "user_prompt": "u", "prompt_version": "v"}
    values[field] = ""
    with pytest.raises(ValidationError):
        LLMRequest(**values)


def test_request_is_frozen():
    with pytest.raises(ValidationError):
        request().user_prompt = "changed"  # type: ignore[misc]


def test_fake_client_is_deterministic_and_records_requests():
    fake = FakeLLMClient("same answer")
    assert isinstance(fake, LLMClient)
    first, second = request("a"), request("b")
    assert fake.generate(first).text == fake.generate(second).text == "same answer"
    assert fake.requests == [first, second] and fake.call_count == 2


def test_fake_client_can_return_a_sequence_and_raise():
    seq = FakeLLMClient(["one", "two"])
    assert [seq.generate(request()).text for _ in range(3)] == ["one", "two", "two"]
    failing = FakeLLMClient(error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        failing.generate(request())
    assert failing.call_count == 1  # recorded before raising


def test_guarded_client_passes_clean_requests_through():
    fake = FakeLLMClient("ok")
    guarded = GuardedLLMClient(fake)
    assert guarded.generate(request()).text == "ok"
    assert fake.call_count == 1
    assert guarded.provider == "fake" and guarded.model == "fake-model"


@pytest.mark.parametrize("where", ["user", "system"])
def test_guarded_client_never_calls_the_inner_client_when_the_gate_raises(where):
    secret = h.github_token("guard")
    fake = FakeLLMClient("ok")
    guarded = GuardedLLMClient(fake)
    req = request(user=f"token = {secret}") if where == "user" else request(system=f"k={secret}")
    with pytest.raises(RepositorySecretRiskError) as info:
        guarded.generate(req)
    assert fake.call_count == 0
    assert not h.contains_secret(str(info.value), secret)


def test_double_wrapping_does_not_nest():
    fake = FakeLLMClient()
    once = GuardedLLMClient(fake)
    twice = GuardedLLMClient(once)
    assert twice._inner is fake  # noqa: SLF001 - the wrapper unwraps so there is one gate
    twice.generate(request())
    assert fake.call_count == 1


def test_guarded_repr_shows_no_secrets():
    assert "fake-model" in repr(GuardedLLMClient(FakeLLMClient()))


def test_factory_requires_a_model_and_a_key_and_never_defaults_a_model():
    with pytest.raises(LLMConfigurationError, match="COPILOT_LLM_MODEL"):
        create_llm_client(Settings(_env_file=None, gemini_api_key="k" * 12))
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        create_llm_client(Settings(_env_file=None, llm_model="some-model"))
    with pytest.raises(LLMConfigurationError, match="not implemented"):
        create_llm_client(Settings(_env_file=None, llm_provider="ollama", llm_model="m"))


def test_factory_returns_a_guarded_client_without_calling_the_network():
    client = create_llm_client(Settings(_env_file=None, llm_model="m-x", gemini_api_key="k" * 12))
    assert isinstance(client, GuardedLLMClient)
    assert client.provider == "gemini" and client.model == "m-x"
    assert "k" * 12 not in repr(client)


def test_settings_defaults_are_conservative_and_have_no_model():
    s = Settings(_env_file=None)
    assert s.llm_model is None
    assert s.llm_temperature == 0.0
    assert s.llm_max_output_tokens == 1024 and s.llm_timeout_seconds == 60.0
    assert s.rag_context_max_tokens == 6000
    key = h.random_value("settings-key", 30)  # generated at run time, not a literal
    configured = Settings(_env_file=None, gemini_api_key=key)
    assert key not in repr(configured) and key not in str(configured)
