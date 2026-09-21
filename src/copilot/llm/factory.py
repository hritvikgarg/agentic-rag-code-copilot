"""Build the configured (and always guarded) LLM client from settings."""

from __future__ import annotations

from copilot.config import Settings, get_settings
from copilot.llm.errors import LLMConfigurationError
from copilot.llm.gemini import GeminiLLMClient
from copilot.llm.guarded import GuardedLLMClient


def create_llm_client(settings: Settings | None = None) -> GuardedLLMClient:
    """The hosted client for ``settings``, wrapped in the secret gate.

    Raises:
        LLMConfigurationError: provider unsupported, model unset or API key unset. The model id is
            never defaulted: choose one from the provider's current documentation.
    """
    settings = settings or get_settings()
    if settings.llm_provider != "gemini":
        raise LLMConfigurationError(
            f"LLM provider {settings.llm_provider!r} is not implemented; "
            "use COPILOT_LLM_PROVIDER=gemini"
        )
    if not settings.llm_model:
        raise LLMConfigurationError(
            "COPILOT_LLM_MODEL is not set. Choose a model id from the provider's current model "
            "list (no default is provided on purpose) and set it in .env"
        )
    return GuardedLLMClient(
        GeminiLLMClient(
            api_key=settings.gemini_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    )
