"""The suite-wide guard that stops any hosted-provider or download attempt at the socket level."""

import socket

import pytest
from pydantic import SecretStr

from copilot.llm import GeminiLLMClient, LLMError, LLMRequest


def test_a_real_gemini_client_cannot_leave_the_host_from_a_test(_no_outbound_network):
    """The REAL client with a dummy key: the call must fail locally and the guard must record it."""
    client = GeminiLLMClient(api_key=SecretStr("x" * 20), model="some-model")
    request = LLMRequest(system_instruction="s", user_prompt="p", prompt_version="v")
    with pytest.raises(LLMError):
        client.generate(request)
    assert any("generativelanguage.googleapis.com" in a for a in _no_outbound_network), (
        "the attempt to reach the provider was not recorded"
    )
    _no_outbound_network.clear()  # expected here; any other test with an attempt fails at teardown


def test_dns_lookup_of_a_remote_host_is_refused_and_recorded(_no_outbound_network):
    with pytest.raises(OSError, match="blocked"):
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    assert _no_outbound_network == ["DNS lookup of 'generativelanguage.googleapis.com'"]
    _no_outbound_network.clear()


def test_an_http_request_is_blocked_even_when_a_proxy_hides_it_from_the_socket_layer(
    _no_outbound_network, monkeypatch
):
    import httpx

    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:3128")
    with pytest.raises(httpx.ConnectError), httpx.Client() as client:
        client.get("https://generativelanguage.googleapis.com/v1/models")
    assert _no_outbound_network == ["HTTP request to 'generativelanguage.googleapis.com'"]
    _no_outbound_network.clear()


def test_loopback_is_still_allowed(_no_outbound_network):
    assert socket.getaddrinfo("127.0.0.1", 80)
    assert _no_outbound_network == []
