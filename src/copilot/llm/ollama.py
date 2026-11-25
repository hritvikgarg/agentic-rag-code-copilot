"""Local Ollama client (Milestone 11B) over plain ``httpx``. No retrieval, RAG or judge logic.

Ollama is a local model server. This client talks to its documented REST API only:

* ``POST /api/chat`` - one non-streaming chat completion (the only call that carries request text),
* ``GET /api/version``, ``GET /api/tags``, ``POST /api/show``, ``GET /api/ps`` - metadata calls that
  carry no repository content (a model name at most), used for the runtime identity and preflight.

Safety properties (each is covered by tests that never open a socket):

* **Loopback only by default.** ``validate_endpoint`` accepts ``localhost`` or a loopback IP
  literal. A remote host needs ``allow_remote=True`` (``COPILOT_OLLAMA_ALLOW_REMOTE``) *and* https.
  The URL may not contain credentials, a path, a query or a fragment. ``localhost`` is connected
  to as ``127.0.0.1`` so a hosts-file or DNS entry cannot redirect it.
* **No redirect or proxy detour.** The HTTP client is built with ``follow_redirects=False`` and
  ``trust_env=False`` (environment proxy variables such as ``HTTP_PROXY`` are ignored), and any
  3xx answer is refused. An injected client that does not have both settings is rejected.
* **Never pulls a model.** A missing model is a ``LLMConfigurationError`` that tells the user the
  command to run; nothing here ever calls ``/api/pull``.
* **Secret gate.** This class does not scan requests itself; the factory only ever hands it out
  wrapped in ``GuardedLLMClient`` (see ``copilot.llm.guarded``), exactly like the Gemini client.
* **Secret-safe errors.** Error messages are fixed strings (host, status code, model name from
  configuration); response bodies and request text are never copied into them and provider
  exceptions are raised ``from None``.

Context safety (``docs/llm.md`` explains the limits honestly):

1. *Before sending*, the input is estimated with a deliberately pessimistic heuristic
   (``CONSERVATIVE_BYTES_PER_TOKEN`` UTF-8 bytes per token plus a fixed template overhead). If that
   estimate exceeds ``num_ctx - max_output_tokens`` the request is refused with
   ``LLMContextLimitError``; nothing is sent, shortened or dropped. The heuristic is not a proof:
   a tokenizer can in principle need more tokens than the estimate.
2. *After the answer*, the provider's metadata is validated. The answer is rejected when
   ``done`` is not true, ``done_reason`` is missing or not ``stop`` (``length`` means the output hit
   its budget), ``prompt_eval_count`` or ``eval_count`` are missing or malformed, the reported
   prompt tokens reach the input budget, the output reaches ``num_predict``, or prompt plus output
   reach ``num_ctx`` (the context may have shifted). ``prompt_eval_count`` alone does NOT prove
   the prompt was complete: with a warm prompt cache Ollama may evaluate (and report) only the
   uncached suffix, and a silently truncated prompt is only detectable when the reported count
   reaches the limit. The checks reduce the risk; they do not remove it.

Reasoning tokens are not reported by Ollama's chat API, so ``reasoning_tokens`` stays ``None``.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import re
import time
from typing import Any, Literal
from urllib.parse import SplitResult, urlsplit

import httpx

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
from copilot.llm.models import LLMRequest, LLMResponse
from copilot.llm.runtime import RuntimeIdentity

logger = logging.getLogger(__name__)

PROVIDER = "ollama"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_NUM_CTX = 8192
DEFAULT_SEED = 20260
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_TIMEOUT_SECONDS = 300.0

# Pessimistic: real English/code text is about 3-4 UTF-8 bytes per token on a Llama 3 tokenizer.
# Counting only 2 bytes per token over-estimates the token count by roughly 1.5-2x, which is the
# safety margin. Non-ASCII text uses more bytes per character and is over-estimated further.
CONSERVATIVE_BYTES_PER_TOKEN = 2.0
# Fixed allowance for the chat template's role markers and special tokens.
TEMPLATE_OVERHEAD_TOKENS = 64

_LOOPBACK_NAMES = {"localhost"}
_KEEP_ALIVE = re.compile(r"^(-1|0|[0-9]+(\.[0-9]+)?(ms|s|m|h))$")
_DIGEST = re.compile(r"^[0-9a-f]{8,}$")

StructuredOutput = Literal["json", "none"]


def estimate_prompt_tokens_upper_bound(system_instruction: str, user_prompt: str) -> int:
    """Pessimistic token estimate for the whole chat prompt (see the module docstring)."""
    size = len(system_instruction.encode("utf-8")) + len(user_prompt.encode("utf-8"))
    return math.ceil(size / CONSERVATIVE_BYTES_PER_TOKEN) + TEMPLATE_OVERHEAD_TOKENS


class Endpoint:
    """A validated Ollama origin. ``connect_url`` is what requests use; ``display`` is for logs."""

    def __init__(self, connect_url: str, display: str, *, loopback: bool) -> None:
        self.connect_url = connect_url
        self.display = display
        self.loopback = loopback

    def __repr__(self) -> str:
        return f"Endpoint({self.display!r})"


def _is_loopback_host(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # not an IP literal (a DNS name, or an ambiguous form such as "127.1")
        return False


def validate_endpoint(base_url: str, *, allow_remote: bool = False) -> Endpoint:
    """Validate ``base_url`` and return the origin requests will use.

    Raises ``LLMConfigurationError`` (with a fixed message that never echoes the URL, since it
    could carry credentials) for anything that is not a plain ``scheme://host[:port]`` origin,
    and for a non-loopback host unless ``allow_remote`` is set (which additionally requires https).
    """
    try:
        parts: SplitResult = urlsplit(base_url.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        raise LLMConfigurationError("the Ollama base URL is not a valid URL") from None
    if parts.scheme not in {"http", "https"}:
        raise LLMConfigurationError("the Ollama base URL must use http or https")
    if not host:
        raise LLMConfigurationError("the Ollama base URL has no host name")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise LLMConfigurationError("the Ollama base URL must not contain credentials")
    if parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise LLMConfigurationError(
            "the Ollama base URL must be a plain origin (no path, query or fragment)"
        )
    if port is not None and not 1 <= port <= 65535:
        raise LLMConfigurationError("the Ollama base URL has an invalid port")
    host = host.lower()
    loopback = _is_loopback_host(host)
    if not loopback:
        if not allow_remote:
            raise LLMConfigurationError(
                "the Ollama endpoint is not a loopback address; repository content would leave "
                "this machine. Use http://localhost:11434, or set "
                "COPILOT_OLLAMA_ALLOW_REMOTE=true to authorise a remote endpoint explicitly"
            )
        if parts.scheme != "https":
            raise LLMConfigurationError("a remote Ollama endpoint must use https")
    connect_host = "127.0.0.1" if host == "localhost" else host
    bracketed = f"[{connect_host}]" if ":" in connect_host else connect_host
    suffix = f":{port}" if port is not None else ""
    connect_url = f"{parts.scheme}://{bracketed}{suffix}"
    display = f"{parts.scheme}://{host if ':' not in host else f'[{host}]'}{suffix}"
    return Endpoint(connect_url, display, loopback=loopback)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _keep_alive_value(keep_alive: str) -> str | int:
    if not _KEEP_ALIVE.match(keep_alive):
        raise LLMConfigurationError(
            "COPILOT_OLLAMA_KEEP_ALIVE must be a duration such as 30m, 0 or -1"
        )
    return int(keep_alive) if re.fullmatch(r"-?[0-9]+", keep_alive) else keep_alive


class OllamaLLMClient:
    """``LLMClient`` for a local Ollama server. ``transport``/``http_client`` let tests inject a
    ``httpx.MockTransport`` so no socket is ever opened."""

    provider = PROVIDER

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        num_ctx: int = DEFAULT_NUM_CTX,
        seed: int = DEFAULT_SEED,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        structured_output: StructuredOutput = "none",
        allow_remote: bool = False,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not model or not model.strip():
            raise LLMConfigurationError("no Ollama model configured (COPILOT_LLM_MODEL)")
        if num_ctx < 256:
            raise LLMConfigurationError("COPILOT_OLLAMA_NUM_CTX is too small to be useful")
        if timeout_seconds <= 0:
            raise LLMConfigurationError("the Ollama timeout must be positive")
        if structured_output not in ("json", "none"):
            raise LLMConfigurationError("structured_output must be 'json' or 'none'")
        if http_client is not None and transport is not None:
            raise LLMConfigurationError("pass either an HTTP client or a transport, not both")
        self.model = model.strip()
        self._endpoint = validate_endpoint(base_url, allow_remote=allow_remote)
        self._timeout = timeout_seconds
        self.num_ctx = num_ctx
        self.seed = seed
        self._keep_alive = _keep_alive_value(keep_alive)
        self.structured_output: StructuredOutput = structured_output
        self._transport = transport
        self._http_client = http_client
        if http_client is not None and (http_client.follow_redirects or http_client.trust_env):
            raise LLMConfigurationError(
                "an injected HTTP client must have follow_redirects=False and trust_env=False"
            )
        self._identity_cache: dict[str, Any] | None = None

    # ------------------------------------------------------------------ transport

    @property
    def endpoint(self) -> Endpoint:
        return self._endpoint

    def _client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(
                transport=self._transport,
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=False,  # a redirect could leave the validated origin
                trust_env=False,  # ignore HTTP(S)_PROXY / ALL_PROXY / NO_PROXY and netrc
            )
        return self._http_client

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def _map_status(self, status: int) -> LLMError:
        if 300 <= status < 400:
            return LLMProviderError(
                f"Ollama answered with a redirect (HTTP {status}); redirects are refused so a "
                "request cannot leave the configured endpoint",
                status_code=status,
            )
        if status in (401, 403):
            return LLMAuthenticationError(
                f"the Ollama endpoint rejected the request (HTTP {status}); a local Ollama server "
                "needs no credentials, so something else is answering on this address"
            )
        if status == 404:
            return LLMConfigurationError(
                "Ollama reports the model (or this endpoint) was not found (HTTP 404). Models are "
                f"never pulled automatically; install it yourself with: ollama pull {self.model}"
            )
        if status == 429:
            return LLMRateLimitError("Ollama is rate limiting requests (HTTP 429)")
        if status in (408, 504):
            return LLMTimeoutError(f"Ollama request timed out (HTTP {status})", status_code=status)
        if status == 400:
            return LLMProviderError("Ollama rejected the request (HTTP 400)", status_code=400)
        return LLMProviderError(f"Ollama request failed (HTTP {status})", status_code=status)

    def _call(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """One HTTP exchange with typed, secret-safe errors; returns the decoded JSON object."""
        url = f"{self._endpoint.connect_url}{path}"
        try:
            response = self._client().request(method, url, json=payload)
        except httpx.TimeoutException:
            raise LLMTimeoutError(
                f"Ollama request timed out after {self._timeout:g}s ({self._endpoint.display})"
            ) from None
        except httpx.TransportError:
            raise LLMConnectionError(
                f"could not reach Ollama at {self._endpoint.display}; is the server running?"
            ) from None
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - never surface the library's message
            raise LLMProviderError(
                f"unexpected Ollama client failure ({type(exc).__name__})"
            ) from None
        if response.status_code != 200:
            raise self._map_status(response.status_code)
        try:
            body = response.json()
        except ValueError:
            raise LLMResponseError("Ollama returned a body that is not valid JSON") from None
        if not isinstance(body, dict):
            raise LLMResponseError("Ollama returned an unexpected JSON shape")
        return body

    # ------------------------------------------------------------------ generation

    def request_body(self, request: LLMRequest) -> dict[str, Any]:
        """The exact JSON sent to ``/api/chat`` (public so tests and docs can show it)."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_instruction},
                {"role": "user", "content": request.user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
                "num_ctx": self.num_ctx,
                "seed": self.seed,
            },
            "keep_alive": self._keep_alive,
        }
        if self.structured_output == "json":
            body["format"] = "json"
        return body

    def input_budget(self, request: LLMRequest) -> int:
        """Prompt tokens allowed: the context window minus the reserved output budget."""
        return self.num_ctx - request.max_output_tokens

    def check_input_fits(self, request: LLMRequest) -> int:
        """Refuse (before any network call) a request whose conservative estimate cannot fit."""
        budget = self.input_budget(request)
        estimate = estimate_prompt_tokens_upper_bound(
            request.system_instruction, request.user_prompt
        )
        if budget <= 0 or estimate > budget:
            raise LLMContextLimitError(
                f"the prompt is estimated at up to {estimate} tokens but only {max(budget, 0)} fit "
                f"(num_ctx {self.num_ctx} minus {request.max_output_tokens} reserved for output); "
                "the request was not sent and nothing was shortened"
            )
        return estimate

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.check_input_fits(request)
        started = time.perf_counter()
        body = self._call("POST", "/api/chat", self.request_body(request))
        elapsed = time.perf_counter() - started
        return self._to_response(body, request, elapsed)

    def _to_response(
        self, body: dict[str, Any], request: LLMRequest, elapsed: float
    ) -> LLMResponse:
        message = body.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("Ollama returned no text")
        if body.get("done") is not True:
            raise LLMResponseError("Ollama did not mark the response as complete")
        done_reason = body.get("done_reason")
        prompt_tokens = _int_or_none(body.get("prompt_eval_count"))
        completion_tokens = _int_or_none(body.get("eval_count"))
        if not isinstance(done_reason, str) or not done_reason:
            raise LLMResponseError("Ollama did not report a completion reason (done_reason)")
        if prompt_tokens is None or completion_tokens is None:
            raise LLMResponseError(
                "Ollama did not report token counts (prompt_eval_count/eval_count); the answer "
                "cannot be checked for truncation"
            )
        self._check_not_truncated(request, done_reason, prompt_tokens, completion_tokens)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            finish_reason=done_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            reasoning_tokens=None,  # Ollama's chat API does not report reasoning tokens
            latency_seconds=round(elapsed, 3),
        )

    def _check_not_truncated(
        self, request: LLMRequest, done_reason: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        budget = self.input_budget(request)
        if done_reason == "length":
            raise LLMContextLimitError(
                "Ollama stopped because the output budget or context was exhausted "
                f"(done_reason=length, {completion_tokens} tokens generated, budget "
                f"{request.max_output_tokens}); the answer is incomplete and was discarded"
            )
        if done_reason != "stop":
            raise LLMResponseError(f"Ollama ended with an unexpected reason ({done_reason!r})")
        if prompt_tokens >= budget:
            raise LLMContextLimitError(
                f"Ollama reports {prompt_tokens} prompt tokens, at or above the input budget of "
                f"{budget} (num_ctx {self.num_ctx}); the prompt may have been truncated"
            )
        if completion_tokens >= request.max_output_tokens:
            raise LLMContextLimitError(
                f"Ollama generated {completion_tokens} tokens, the whole output budget "
                f"({request.max_output_tokens}); the answer may be cut off"
            )
        if prompt_tokens + completion_tokens >= self.num_ctx:
            raise LLMContextLimitError(
                f"prompt ({prompt_tokens}) plus output ({completion_tokens}) tokens reach "
                f"num_ctx {self.num_ctx}; the context may have shifted during generation"
            )

    # ------------------------------------------------------------------ metadata (no request text)

    def server_version(self) -> str:
        """``GET /api/version`` - the Ollama server version string."""
        version = self._call("GET", "/api/version").get("version")
        if not isinstance(version, str) or not version.strip():
            raise LLMResponseError("Ollama did not report its version")
        return version.strip()

    def list_models(self) -> list[dict[str, Any]]:
        """``GET /api/tags`` - the locally installed models."""
        models = self._call("GET", "/api/tags").get("models")
        if not isinstance(models, list):
            raise LLMResponseError("Ollama returned no model list")
        return [m for m in models if isinstance(m, dict)]

    def running_models(self) -> list[dict[str, Any]]:
        """``GET /api/ps`` - the models currently loaded (and how much of each is on the GPU)."""
        models = self._call("GET", "/api/ps").get("models")
        if not isinstance(models, list):
            raise LLMResponseError("Ollama returned no running-model list")
        return [m for m in models if isinstance(m, dict)]

    def installed_model(self) -> dict[str, Any]:
        """The ``/api/tags`` entry for ``self.model`` (matching ``name`` or ``name:latest``)."""
        wanted = {self.model, f"{self.model}:latest" if ":" not in self.model else self.model}
        for entry in self.list_models():
            if entry.get("name") in wanted or entry.get("model") in wanted:
                return entry
        raise LLMConfigurationError(
            f"the model is not installed locally. Models are never pulled automatically; "
            f"install it yourself with: ollama pull {self.model}"
        )

    def model_details(self) -> dict[str, Any]:
        """Digest, quantisation and parameter size read from the runtime (never assumed)."""
        entry = self.installed_model()
        digest = entry.get("digest")
        details = entry.get("details")
        if not isinstance(details, dict):
            shown = self._call("POST", "/api/show", {"model": self.model})
            details = shown.get("details")
        if not isinstance(details, dict):
            raise LLMResponseError("Ollama did not report model details")
        quantization = details.get("quantization_level")
        size = details.get("parameter_size")
        if not (isinstance(digest, str) and _DIGEST.match(digest.lower())):
            raise LLMResponseError("Ollama did not report a valid model digest")
        if not isinstance(quantization, str) or not quantization:
            raise LLMResponseError("Ollama did not report the model quantisation")
        if not isinstance(size, str) or not size:
            raise LLMResponseError("Ollama did not report the model parameter size")
        return {
            "name": entry.get("name") or self.model,
            "digest": digest.lower(),
            "quantization": quantization,
            "parameter_size": size,
            "family": details.get("family"),
            "format": details.get("format"),
        }

    def describe_runtime(self) -> dict[str, Any]:
        """Metadata-only report for ``judge-preflight``: server, model and GPU residency.

        ``loaded`` is ``None`` when the model is not currently in memory (warm it up and look
        again) or when ``/api/ps`` is unavailable on this server version. Nothing is pulled and no
        request text is sent.
        """
        version = self.server_version()
        details = self.model_details()
        loaded: dict[str, Any] | None = None
        ps_note: str | None = None
        try:
            running = self.running_models()
        except LLMError:
            running = []
            ps_note = "the server did not answer /api/ps; GPU residency is unknown"
        for entry in running:
            if details["name"] in (entry.get("name"), entry.get("model")):
                size, vram = entry.get("size"), entry.get("size_vram")
                ok = isinstance(size, int) and isinstance(vram, int) and size > 0
                loaded = {
                    "size_bytes": size if isinstance(size, int) else None,
                    "size_vram_bytes": vram if isinstance(vram, int) else None,
                    "gpu_fraction": round(vram / size, 4) if ok else None,
                    "context_length": _int_or_none(entry.get("context_length")),
                    "expires_at": entry.get("expires_at"),
                }
        return {
            "endpoint": self._endpoint.display,
            "loopback": self._endpoint.loopback,
            "ollama_version": version,
            "model": details,
            "loaded": loaded,
            "notes": [ps_note] if ps_note else [],
        }

    def runtime_identity(self, *, temperature: float, max_output_tokens: int) -> RuntimeIdentity:
        """The full identity of this runtime for a given sampling/output setting.

        Model details and the server version are read once per client and then reused, so a model
        replaced under a running process is only noticed by the next process (documented).
        """
        if self._identity_cache is None:
            details = self.model_details()
            details["runtime_version"] = self.server_version()
            self._identity_cache = details
        info = self._identity_cache
        return RuntimeIdentity(
            provider=self.provider,
            model_tag=self.model,
            model_digest=info["digest"],
            quantization=info["quantization"],
            parameter_size=info["parameter_size"],
            runtime_version=info["runtime_version"],
            context_size=self.num_ctx,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            seed=self.seed,
            structured_output=self.structured_output,
        )

    def __repr__(self) -> str:
        return f"OllamaLLMClient(model={self.model!r}, endpoint={self._endpoint.display!r})"
