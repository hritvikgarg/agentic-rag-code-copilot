"""Google Gemini client (official ``google-genai`` SDK). Contains no retrieval or RAG logic.

Verified against google-genai 2.24.0: ``genai.Client(api_key=..., http_options=HttpOptions(timeout=
<milliseconds>))``, ``client.models.generate_content(model=, contents=, config=
GenerateContentConfig(system_instruction=, temperature=, max_output_tokens=))``, ``response.text``,
``response.usage_metadata`` and ``response.candidates[0].finish_reason``.

No model id is hard-coded: the model comes from ``COPILOT_LLM_MODEL``. The API key is held as a
``SecretStr`` and only unwrapped when the SDK client is created. Provider exceptions are mapped
to typed errors with fixed messages and re-raised ``from None`` (the SDK's own message can echo
request content).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import SecretStr

from copilot.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from copilot.llm.models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


def _label(value: Any) -> str | None:
    """A finish reason / block reason as a plain string (enum name or str)."""
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def map_api_error(code: int | None) -> LLMError:
    """Turn an HTTP status from the provider into a typed error with a fixed message."""
    if code in (401, 403):
        return LLMAuthenticationError(
            f"Gemini rejected the credentials (HTTP {code}); check GEMINI_API_KEY"
        )
    if code == 404:
        return LLMConfigurationError(
            "Gemini reports the model was not found or is not available to this key (HTTP 404); "
            "check COPILOT_LLM_MODEL against the provider's current model list"
        )
    if code == 429:
        return LLMRateLimitError("Gemini rate or quota limit reached (HTTP 429)")
    if code in (408, 504):
        return LLMTimeoutError(f"Gemini request timed out (HTTP {code})")
    if code == 400:
        return LLMProviderError(
            "Gemini rejected the request (HTTP 400: invalid request, or an invalid API key)",
            status_code=400,
        )
    suffix = f" (HTTP {code})" if code else ""
    return LLMProviderError(f"Gemini request failed{suffix}", status_code=code)


class GeminiLLMClient:
    """``LLMClient`` for the Gemini API. ``sdk_client`` exists so tests can inject a fake."""

    provider = "gemini"

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str,
        timeout_seconds: float = 60.0,
        sdk_client: Any | None = None,
    ) -> None:
        if not model:
            raise LLMConfigurationError("no Gemini model configured (COPILOT_LLM_MODEL)")
        if sdk_client is None and (api_key is None or not api_key.get_secret_value()):
            raise LLMConfigurationError("GEMINI_API_KEY is not set")
        self.model = model
        self._api_key = api_key
        self._timeout_ms = max(1, int(timeout_seconds * 1000))
        self._sdk_client = sdk_client

    def _client(self) -> Any:
        if self._sdk_client is None:
            from google import genai

            assert self._api_key is not None
            try:
                self._sdk_client = genai.Client(
                    api_key=self._api_key.get_secret_value(),
                    http_options=genai_types.HttpOptions(timeout=self._timeout_ms),
                )
            except Exception:  # noqa: BLE001 - the SDK message could contain the key
                raise LLMConfigurationError("could not create the Gemini client") from None
        return self._sdk_client

    def generate(self, request: LLMRequest) -> LLMResponse:
        client = self._client()
        config = genai_types.GenerateContentConfig(
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )
        started = time.perf_counter()
        try:
            raw = client.models.generate_content(
                model=self.model, contents=request.user_prompt, config=config
            )
        except genai_errors.APIError as exc:
            raise map_api_error(getattr(exc, "code", None)) from None
        except httpx.TimeoutException:
            raise LLMTimeoutError("Gemini request timed out") from None
        except httpx.HTTPError:
            raise LLMConnectionError("could not reach the Gemini API") from None
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - never surface the SDK's message
            raise LLMProviderError(
                f"unexpected Gemini client failure ({type(exc).__name__})"
            ) from None
        elapsed = time.perf_counter() - started
        return self._to_response(raw, elapsed)

    def _to_response(self, raw: Any, elapsed: float) -> LLMResponse:
        feedback = getattr(raw, "prompt_feedback", None)
        block = _label(getattr(feedback, "block_reason", None))
        candidates = getattr(raw, "candidates", None) or []
        finish = _label(getattr(candidates[0], "finish_reason", None)) if candidates else None
        try:
            text = getattr(raw, "text", None)
        except Exception:  # noqa: BLE001 - the SDK may raise for non-text parts
            text = None
        if block:
            raise LLMResponseError(f"Gemini blocked the prompt (reason: {block})")
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError(
                "Gemini returned no text"
                + (f" (finish reason: {finish})" if finish else "")
                + "; a MAX_TOKENS reason means COPILOT_LLM_MAX_OUTPUT_TOKENS is too small"
            )
        usage = getattr(raw, "usage_metadata", None)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            finish_reason=finish,
            prompt_tokens=_int_or_none(getattr(usage, "prompt_token_count", None)),
            completion_tokens=_int_or_none(getattr(usage, "candidates_token_count", None)),
            total_tokens=_int_or_none(getattr(usage, "total_token_count", None)),
            latency_seconds=round(elapsed, 3),
        )

    def __repr__(self) -> str:
        return f"GeminiLLMClient(model={self.model!r})"  # never includes the key
