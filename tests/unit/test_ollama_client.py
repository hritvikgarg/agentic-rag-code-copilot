"""Milestone 11B: the local Ollama client, exercised only through ``httpx.MockTransport``.

No socket is opened (the autouse network guard would fail the test otherwise) and no Ollama
server, model or API key is involved.
"""

from __future__ import annotations

import json

import httpx
import pytest

from copilot.config import Settings
from copilot.llm import (
    GuardedLLMClient,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMContextLimitError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseError,
    LLMTimeoutError,
    OllamaLLMClient,
    create_llm_client,
)
from copilot.llm.ollama import (
    TEMPLATE_OVERHEAD_TOKENS,
    estimate_prompt_tokens_upper_bound,
    validate_endpoint,
)
from copilot.llm.retry import is_transient
from copilot.security import RepositorySecretRiskError
from tests.security import secret_helpers as h

DIGEST = "a" * 64


def request(system="SYS", user="USER question", **kw) -> LLMRequest:
    kw.setdefault("max_output_tokens", 1024)
    return LLMRequest(system_instruction=system, user_prompt=user, prompt_version="t-1", **kw)


def chat_ok(content='{"a": 1}', **overrides):
    body = {
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 500,
        "eval_count": 40,
    }
    body.update(overrides)
    return body


class Server:
    """A scripted fake Ollama: records every request and answers by path."""

    def __init__(self, chat=None, tags=None, version="0.12.3", show=None, status=200):
        self.calls: list[httpx.Request] = []
        self.chat = chat if chat is not None else chat_ok()
        self.tags = tags if tags is not None else installed()
        self.version = version
        self.status = status

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "SECRET-BODY-TEXT"})
        path = req.url.path
        if path == "/api/chat":
            return httpx.Response(200, json=self.chat)
        if path == "/api/tags":
            return httpx.Response(200, json=self.tags)
        if path == "/api/show":
            return httpx.Response(200, json={})  # no details: forces the "incomplete" branch
        if path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        return httpx.Response(404, json={"error": "unknown"})

    @property
    def paths(self) -> list[str]:
        return [c.url.path for c in self.calls]


def installed(digest=DIGEST, quant="Q4_K_M", size="8.0B", name="llama3.1:8b"):
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "digest": digest,
                "details": {"quantization_level": quant, "parameter_size": size, "family": "llama"},
            }
        ]
    }


def make(server: Server | None = None, **kw) -> tuple[OllamaLLMClient, Server]:
    server = server or Server()
    kw.setdefault("model", "llama3.1:8b")
    return OllamaLLMClient(transport=httpx.MockTransport(server), **kw), server


# ---------------------------------------------------------------- request shape / response map


def test_request_shape_is_explicit_and_complete():
    client, server = make(structured_output="json", seed=7, keep_alive="10m", num_ctx=4096)
    client.generate(request(temperature=0.0, max_output_tokens=256))
    (call,) = server.calls
    assert (call.method, call.url.path) == ("POST", "/api/chat")
    body = json.loads(call.content)
    assert body == {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER question"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 256, "num_ctx": 4096, "seed": 7},
        "keep_alive": "10m",
    }


def test_text_mode_sends_no_format_and_numeric_keep_alive_is_an_integer():
    client, server = make(keep_alive="0")
    client.generate(request())
    body = json.loads(server.calls[0].content)
    assert "format" not in body and body["keep_alive"] == 0


@pytest.mark.parametrize("bad", ["", "soon", "5 minutes", "1d", "-2"])
def test_keep_alive_is_validated(bad):
    with pytest.raises(LLMConfigurationError, match="KEEP_ALIVE"):
        OllamaLLMClient(model="m", keep_alive=bad, transport=httpx.MockTransport(Server()))


def test_response_mapping_and_metadata():
    client, _ = make(Server(chat=chat_ok("hello", prompt_eval_count=321, eval_count=17)))
    response = client.generate(request())
    assert response.text == "hello"
    assert (response.provider, response.model) == ("ollama", "llama3.1:8b")
    assert response.finish_reason == "stop"
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (
        321,
        17,
        338,
    )
    assert response.reasoning_tokens is None
    assert response.latency_seconds is not None and response.latency_seconds >= 0


def test_localhost_is_connected_to_as_loopback_ip():
    client, server = make(base_url="http://localhost:11434")
    client.generate(request())
    assert server.calls[0].url.host == "127.0.0.1" and server.calls[0].url.port == 11434


# ---------------------------------------------------------------- error mapping


@pytest.mark.parametrize(
    ("status", "exc"),
    [
        (401, LLMAuthenticationError),
        (403, LLMAuthenticationError),
        (404, LLMConfigurationError),
        (429, LLMRateLimitError),
        (408, LLMTimeoutError),
        (504, LLMTimeoutError),
        (400, LLMProviderError),
        (500, LLMProviderError),
        (503, LLMProviderError),
    ],
)
def test_http_status_mapping_never_leaks_the_body_or_the_prompt(status, exc):
    client, server = make(Server(status=status))
    with pytest.raises(exc) as info:
        client.generate(request(user="PRIVATE-PROMPT-TEXT"))
    message = str(info.value)
    assert "SECRET-BODY-TEXT" not in message and "PRIVATE-PROMPT-TEXT" not in message
    assert server.paths == ["/api/chat"]  # never a pull or any other path


def test_missing_model_message_says_how_to_pull_but_nothing_is_pulled():
    client, server = make(Server(status=404))
    with pytest.raises(LLMConfigurationError, match="ollama pull llama3.1:8b"):
        client.generate(request())
    assert all("pull" not in p for p in server.paths)


def test_503_is_transient_and_504_is_transient_but_a_client_timeout_is_not():
    client503, _ = make(Server(status=503))
    with pytest.raises(LLMProviderError) as e503:
        client503.generate(request())
    assert e503.value.status_code == 503 and is_transient(e503.value)

    def timeout(_req):
        raise httpx.ReadTimeout("slow", request=_req)

    client = OllamaLLMClient(model="m", transport=httpx.MockTransport(timeout))
    with pytest.raises(LLMTimeoutError) as info:
        client.generate(request())
    assert info.value.status_code is None and not is_transient(info.value)


def test_connection_failure_is_a_typed_error_naming_only_the_endpoint():
    def refuse(req):
        raise httpx.ConnectError("boom PRIVATE-PROMPT-TEXT", request=req)

    client = OllamaLLMClient(model="m", transport=httpx.MockTransport(refuse))
    with pytest.raises(LLMConnectionError, match="127.0.0.1:11434|localhost:11434") as info:
        client.generate(request(user="PRIVATE-PROMPT-TEXT"))
    assert "PRIVATE-PROMPT-TEXT" not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_non_json_and_non_object_bodies_are_response_errors():
    def not_json(_req):
        return httpx.Response(200, content=b"<html>")

    def a_list(_req):
        return httpx.Response(200, json=[1])

    for handler in (not_json, a_list):
        client = OllamaLLMClient(model="m", transport=httpx.MockTransport(handler))
        with pytest.raises(LLMResponseError):
            client.generate(request())


@pytest.mark.parametrize(
    "chat",
    [
        chat_ok(content=""),
        chat_ok(content="   "),
        {**chat_ok(), "message": {"role": "assistant"}},
        chat_ok(done=False),
        {k: v for k, v in chat_ok().items() if k != "done_reason"},
        {k: v for k, v in chat_ok().items() if k != "prompt_eval_count"},
        {k: v for k, v in chat_ok().items() if k != "eval_count"},
        chat_ok(prompt_eval_count="many"),
        chat_ok(eval_count=-3),
        chat_ok(eval_count=True),
        chat_ok(done_reason="load"),
    ],
)
def test_malformed_or_incomplete_metadata_is_rejected(chat):
    client, _ = make(Server(chat=chat))
    with pytest.raises(LLMResponseError):
        client.generate(request())


# ---------------------------------------------------------------- endpoint security


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://localhost",
        "http://127.0.0.1:11434/",
        "http://127.5.5.5:11434",
        "http://[::1]:11434",
        "https://localhost:11434",
        "HTTP://LOCALHOST:11434",
    ],
)
def test_loopback_endpoints_are_accepted(url):
    assert validate_endpoint(url).loopback


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:11434",
        "http://192.168.1.20:11434",
        "http://10.0.0.5",
        "http://0.0.0.0:11434",
        "http://localhost.evil.com:11434",
        "http://127.0.0.1.evil.com",
        "http://127.1:11434",
        "http://2130706433:11434",
        "http://[::ffff:10.0.0.1]:11434",
        "ftp://localhost:11434",
        "file:///etc/passwd",
        "localhost:11434",
        "//localhost:11434",
        "http://user:pw@localhost:11434",
        "http://localhost@evil.com:11434",
        "http://localhost:11434/api",
        "http://localhost:11434?x=1",
        "http://localhost:11434#frag",
        "http://localhost:99999",
        "http://",
        "",
    ],
)
def test_everything_else_is_refused_without_echoing_the_url(url):
    with pytest.raises(LLMConfigurationError) as info:
        validate_endpoint(url)
    assert "pw@" not in str(info.value) and "evil" not in str(info.value)


def test_remote_needs_explicit_authorisation_and_https():
    assert not validate_endpoint("https://gpu.example.com:11434", allow_remote=True).loopback
    with pytest.raises(LLMConfigurationError, match="https"):
        validate_endpoint("http://gpu.example.com:11434", allow_remote=True)
    with pytest.raises(LLMConfigurationError, match="loopback"):
        validate_endpoint("https://gpu.example.com:11434", allow_remote=False)


def test_a_redirect_is_refused_and_never_followed():
    seen: list[str] = []

    def redirect(req):
        seen.append(str(req.url))
        return httpx.Response(302, headers={"Location": "http://10.9.9.9:11434/api/chat"})

    client = OllamaLLMClient(model="m", transport=httpx.MockTransport(redirect))
    with pytest.raises(LLMProviderError, match="redirect"):
        client.generate(request())
    assert len(seen) == 1 and "10.9.9.9" not in seen[0]


def test_environment_proxies_are_ignored_and_redirects_disabled(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://proxy.invalid:3128")
    client, _ = make()
    http = client._client()  # noqa: SLF001
    assert http.trust_env is False and http.follow_redirects is False
    assert http._mounts == {}  # noqa: SLF001 - no proxy transport was mounted


@pytest.mark.parametrize("flags", [{"follow_redirects": True}, {"trust_env": True}])
def test_an_injected_client_must_not_follow_redirects_or_trust_the_environment(flags):
    kwargs = {
        "transport": httpx.MockTransport(Server()),
        "follow_redirects": False,
        "trust_env": False,
    }
    kwargs.update(flags)
    with pytest.raises(LLMConfigurationError, match="follow_redirects=False"):
        OllamaLLMClient(model="m", http_client=httpx.Client(**kwargs))


def test_a_safe_injected_client_is_used():
    server = Server()
    http = httpx.Client(
        transport=httpx.MockTransport(server), follow_redirects=False, trust_env=False
    )
    OllamaLLMClient(model="m", http_client=http).generate(request())
    assert server.paths == ["/api/chat"]


# ---------------------------------------------------------------- context safety


def _prompt_of(estimate_tokens: int) -> tuple[str, str]:
    """A (system, user) pair whose conservative estimate is exactly ``estimate_tokens``."""
    size = (estimate_tokens - TEMPLATE_OVERHEAD_TOKENS) * 2
    return "s" * 10, "u" * (size - 10)


def test_estimator_is_pessimistic_and_counts_utf8_bytes():
    assert estimate_prompt_tokens_upper_bound("ab", "cd") == 2 + TEMPLATE_OVERHEAD_TOKENS
    assert estimate_prompt_tokens_upper_bound("é", "") == 1 + TEMPLATE_OVERHEAD_TOKENS  # 2 bytes


def test_input_exactly_at_the_budget_is_sent_and_one_token_over_is_refused():
    budget = 8192 - 1024
    client, server = make()
    system, user = _prompt_of(budget)
    assert estimate_prompt_tokens_upper_bound(system, user) == budget
    client.generate(request(system, user))
    assert len(server.calls) == 1

    system, user = _prompt_of(budget)
    with pytest.raises(LLMContextLimitError, match="not sent"):
        client.generate(request(system, user + "xx"))
    assert len(server.calls) == 1  # the refused request never reached the transport


def test_the_input_is_never_shortened():
    client, server = make()
    system, user = _prompt_of(2000)
    client.generate(request(system, user))
    sent = json.loads(server.calls[0].content)["messages"]
    assert sent[0]["content"] == system and sent[1]["content"] == user


def test_an_output_budget_as_large_as_the_context_leaves_no_room_for_input():
    client, server = make(num_ctx=2048)
    with pytest.raises(LLMContextLimitError):
        client.generate(request(max_output_tokens=2048))
    assert server.calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"done_reason": "length"},
        {"prompt_eval_count": 7168},  # == input budget: the prompt may have been cut
        {"prompt_eval_count": 8192},
        {"eval_count": 1024},  # the whole output budget used
        {"prompt_eval_count": 7000, "eval_count": 1192},  # together reach num_ctx
    ],
)
def test_responses_that_may_be_truncated_are_rejected(overrides):
    client, _ = make(Server(chat=chat_ok(**overrides)))
    with pytest.raises(LLMContextLimitError):
        client.generate(request(max_output_tokens=1024))


def test_just_below_every_limit_is_accepted():
    client, _ = make(Server(chat=chat_ok(prompt_eval_count=7167, eval_count=1023)))
    # 7167 + 1023 = 8190 < 8192; prompt below 7168; output below 1024
    assert client.generate(request(max_output_tokens=1024)).total_tokens == 8190


def test_a_context_failure_is_not_retried_and_is_an_llm_error():
    assert not is_transient(LLMContextLimitError("x"))


# ---------------------------------------------------------------- runtime identity


def test_runtime_identity_is_read_from_the_runtime_and_never_pulls():
    client, server = make()
    identity = client.runtime_identity(temperature=0.0, max_output_tokens=1024)
    assert identity.model_digest == DIGEST and identity.quantization == "Q4_K_M"
    assert identity.parameter_size == "8.0B" and identity.runtime_version == "0.12.3"
    assert (identity.context_size, identity.seed, identity.structured_output) == (
        8192,
        20260,
        "none",
    )
    assert sorted(server.paths) == ["/api/tags", "/api/version"]
    client.runtime_identity(temperature=0.0, max_output_tokens=512)  # reuses the metadata
    assert len(server.calls) == 2


def test_identity_for_a_model_that_is_not_installed_says_to_pull_it_manually():
    client, server = make(Server(tags={"models": []}))
    with pytest.raises(LLMConfigurationError, match="ollama pull"):
        client.runtime_identity(temperature=0.0, max_output_tokens=1024)
    assert all("pull" not in p for p in server.paths)


@pytest.mark.parametrize(
    "tags",
    [
        installed(digest="short"),
        installed(digest=""),
        installed(quant=""),
        installed(size=""),
        {"models": [{"name": "llama3.1:8b", "digest": DIGEST}]},
    ],
)
def test_incomplete_runtime_metadata_refuses_to_build_an_identity(tags):
    client, _ = make(Server(tags=tags))
    with pytest.raises(LLMResponseError):
        client.runtime_identity(temperature=0.0, max_output_tokens=1024)


def _fp(server=None, *, temperature=0.0, out=1024, **kw):
    client, _ = make(server, **kw)
    return client.runtime_identity(temperature=temperature, max_output_tokens=out).fingerprint()


def test_the_fingerprint_is_stable_and_sensitive_to_every_field():
    base = _fp()
    assert base == _fp() and len(base) == 64
    variants = {
        "digest": _fp(Server(tags=installed(digest="b" * 64))),
        "quant": _fp(Server(tags=installed(quant="Q8_0"))),
        "size": _fp(Server(tags=installed(size="70B"))),
        "version": _fp(Server(version="0.13.0")),
        "num_ctx": _fp(num_ctx=4096),
        "seed": _fp(seed=1),
        "temperature": _fp(temperature=0.5),
        "output": _fp(out=512),
        "structured": _fp(structured_output="json"),
    }
    assert base not in variants.values()
    assert len(set(variants.values())) == len(variants)


def test_keep_alive_does_not_change_the_fingerprint():
    assert _fp(keep_alive="30m") == _fp(keep_alive="0")


def test_the_fingerprint_has_a_golden_value():
    # Frozen on purpose: a change here means every cached local judgement is invalidated.
    client, _ = make()
    identity = client.runtime_identity(temperature=0.0, max_output_tokens=1024)
    assert identity.canonical_json() == (
        '{"context_size":8192,"max_output_tokens":1024,"model_digest":"' + DIGEST + '",'
        '"model_tag":"llama3.1:8b","parameter_size":"8.0B","provider":"ollama",'
        '"quantization":"Q4_K_M","runtime_version":"0.12.3","seed":20260,'
        '"structured_output":"none","temperature":0.0}'
    )


# ---------------------------------------------------------------- factory and the guard


def test_the_factory_builds_a_guarded_ollama_client_without_any_network_call():
    settings = Settings(_env_file=None, llm_provider="ollama", llm_model="llama3.1:8b")
    client = create_llm_client(settings, structured_output="json")
    assert isinstance(client, GuardedLLMClient)
    inner = client._inner  # noqa: SLF001
    assert isinstance(inner, OllamaLLMClient) and inner.structured_output == "json"
    assert (inner.num_ctx, inner.seed) == (8192, 20260)


def test_the_factory_refuses_a_remote_endpoint_unless_authorised():
    base = {
        "llm_provider": "ollama",
        "llm_model": "m",
        "ollama_base_url": "https://gpu.example.com",
    }
    with pytest.raises(LLMConfigurationError, match="loopback"):
        create_llm_client(Settings(_env_file=None, **base))
    create_llm_client(Settings(_env_file=None, ollama_allow_remote=True, **base))


def test_the_guard_blocks_a_secret_before_the_ollama_client_is_called():
    client, server = make()
    guarded = GuardedLLMClient(client)
    with pytest.raises(RepositorySecretRiskError):
        guarded.generate(request(user=f"token {h.github_token()}"))
    assert server.calls == []
    guarded.generate(request())
    assert server.paths == ["/api/chat"]


def test_the_guard_exposes_the_runtime_identity_only_for_local_clients():
    from copilot.llm import FakeLLMClient

    client, _ = make()
    identity = GuardedLLMClient(client).runtime_identity(temperature=0.0, max_output_tokens=1024)
    assert identity is not None and identity.provider == "ollama"
    assert (
        GuardedLLMClient(FakeLLMClient()).runtime_identity(temperature=0.0, max_output_tokens=1)
        is None
    )


def test_repr_has_no_request_text():
    client, _ = make()
    assert "SYS" not in repr(client) and "llama3.1:8b" in repr(client)
