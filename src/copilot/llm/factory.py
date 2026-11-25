"""Build the configured (and always guarded) LLM client from settings."""

from __future__ import annotations

from copilot.config import Settings, get_settings
from copilot.llm.errors import LLMConfigurationError
from copilot.llm.gemini import GeminiLLMClient
from copilot.llm.guarded import GuardedLLMClient
from copilot.llm.ollama import OllamaLLMClient, StructuredOutput


def create_llm_client(
    settings: Settings | None = None, *, structured_output: StructuredOutput = "none"
) -> GuardedLLMClient:
    """The configured client for ``settings``, wrapped in the secret gate.

    ``structured_output="json"`` asks a local (Ollama) model for JSON-mode output (used by the
    judge); it has no effect on Gemini, whose behaviour is unchanged.

    Raises:
        LLMConfigurationError: provider unsupported, model unset or API key unset. The model id is
            never defaulted: choose one from the provider's current documentation.
    """
    settings = settings or get_settings()
    if not settings.llm_model:
        raise LLMConfigurationError(
            "COPILOT_LLM_MODEL is not set. Choose a model id from the provider's current model "
            "list (no default is provided on purpose) and set it in .env"
        )
    if settings.llm_provider == "ollama":
        return GuardedLLMClient(
            OllamaLLMClient(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                timeout_seconds=settings.llm_timeout_seconds,
                num_ctx=settings.ollama_num_ctx,
                seed=settings.ollama_seed,
                keep_alive=settings.ollama_keep_alive,
                structured_output=structured_output,
                allow_remote=settings.ollama_allow_remote,
            )
        )
    if settings.llm_provider != "gemini":
        raise LLMConfigurationError(
            f"LLM provider {settings.llm_provider!r} is not implemented; "
            "use COPILOT_LLM_PROVIDER=gemini or ollama"
        )
    return GuardedLLMClient(
        GeminiLLMClient(
            api_key=settings.gemini_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    )
