"""Application settings loaded from environment variables and an optional ``.env`` file.

Why this exists: every tunable value (model names, chunk sizes, limits, log level) lives in one
typed, validated place instead of being scattered as hard-coded constants. Invalid values fail
at startup with a clear message. Secrets are read from the environment only and are never
printed (``SecretStr``).

Non-secret variables use the ``COPILOT_`` prefix (e.g. ``COPILOT_RETRIEVAL_TOP_K``). The Gemini
key uses the conventional ``GEMINI_API_KEY`` name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProvider = Literal["gemini", "ollama"]
ChunkingStrategy = Literal["line", "ast"]


class Settings(BaseSettings):
    """Typed configuration for the whole application.

    Only values that later milestones will actually consume are defined here; no component
    reads them yet (Milestone 1 is infrastructure only).
    """

    model_config = SettingsConfigDict(
        env_prefix="COPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # --- LLM ---
    llm_provider: LLMProvider = "gemini"
    # Deliberately unset: the concrete model ID is chosen and verified in Milestone 6 (against the
    # provider's current docs and the account's quota). No unverified ID is stored as a default.
    llm_model: str | None = Field(default=None, min_length=1)
    # Conservative generation defaults (Milestone 6). temperature 0 keeps the plain-vs-RAG
    # comparison as repeatable as the provider allows.
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=1024, ge=16, le=65536)
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    # Budget for the repository evidence placed in one RAG prompt, in *estimated* tokens
    # (the same heuristic as chunking). Whole evidence blocks are dropped, never cut in half.
    rag_context_max_tokens: int = Field(default=6000, ge=200, le=200_000)
    ollama_base_url: str = "http://localhost:11434"
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "COPILOT_GEMINI_API_KEY"),
    )

    # --- Embeddings ---
    # Preferred model; subject to the Milestone 4 compatibility test (fallback: bge-small).
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)  # texts per ONNX forward pass
    embedding_threads: int | None = Field(default=None, ge=1)  # None: onnxruntime decides
    # Where downloaded model files are cached. None -> <data_dir>/cache/models (git-ignored).
    # fastembed's own default is the OS temp directory, which can be purged, so we set our own.
    embedding_cache_dir: Path | None = None
    # How a chunk is turned into embedding text (see copilot.embeddings.representation).
    embedding_text_style: Literal["prefixed", "raw"] = "prefixed"

    # --- Chunking and retrieval (initial defaults, to be tuned by measurement) ---
    chunking_strategy: ChunkingStrategy = "line"
    chunk_size_lines: int = Field(default=60, ge=5, le=1000)
    chunk_overlap_lines: int = Field(default=10, ge=0)
    # Hard cap (heuristic token estimate, see copilot.utils.tokens) so no chunk can exceed the
    # embedding model's input limit. 512 is safe for the smallest planned model (bge-small).
    chunk_max_tokens: int = Field(default=512, ge=16, le=8192)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)

    # --- Repository ingestion limits ---
    max_file_size_bytes: int = Field(default=500_000, ge=1_000)
    max_repo_files: int = Field(default=10_000, ge=1)
    max_repo_total_mb: int = Field(default=200, ge=1)

    # --- Runtime ---
    log_level: LogLevel = "INFO"
    data_dir: Path = Path("data")

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Settings:
        if self.chunk_overlap_lines >= self.chunk_size_lines:
            raise ValueError(
                f"chunk_overlap_lines ({self.chunk_overlap_lines}) must be smaller than "
                f"chunk_size_lines ({self.chunk_size_lines})"
            )
        return self

    @property
    def indexes_dir(self) -> Path:
        """Where built vector indexes will live (git-ignored)."""
        return self.data_dir / "indexes"

    @property
    def repos_dir(self) -> Path:
        """Where cloned repositories will live (git-ignored)."""
        return self.data_dir / "repos"

    @property
    def uploads_dir(self) -> Path:
        """Where uploaded archives will be extracted (git-ignored)."""
        return self.data_dir / "uploads"

    @property
    def cache_dir(self) -> Path:
        """Where response/model caches will live (git-ignored)."""
        return self.data_dir / "cache"

    @property
    def model_cache_dir(self) -> Path:
        """Where embedding model files are cached (git-ignored unless overridden)."""
        return self.embedding_cache_dir or self.cache_dir / "models"

    def secret_values(self) -> list[str]:
        """Plain-text secret values, used only to redact them from log output."""
        secrets: list[str] = []
        if self.gemini_api_key is not None:
            secrets.append(self.gemini_api_key.get_secret_value())
        return secrets


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (loaded once; call ``cache_clear()`` in tests)."""
    return Settings()
