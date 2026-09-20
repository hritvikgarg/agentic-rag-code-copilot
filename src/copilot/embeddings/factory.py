"""Create the configured embedder from ``Settings``."""

from __future__ import annotations

from copilot.config.settings import Settings, get_settings
from copilot.embeddings.fastembed_embedder import FastEmbedEmbedder


def create_embedder(settings: Settings | None = None) -> FastEmbedEmbedder:
    """Build the fastembed embedder described by ``settings`` (the model loads on first use)."""
    settings = settings or get_settings()
    return FastEmbedEmbedder(
        settings.embedding_model,
        cache_dir=settings.model_cache_dir,
        batch_size=settings.embedding_batch_size,
        threads=settings.embedding_threads,
    )
