"""Embeddings behind a small interface (Milestone 4). Local models; no query-time network."""

from copilot.embeddings.base import Embedder, TokenCountingEmbedder
from copilot.embeddings.chunk_embeddings import ChunkEmbeddings, embed_chunks
from copilot.embeddings.errors import (
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingModelUnavailableError,
    EmbeddingRuntimeError,
)
from copilot.embeddings.factory import create_embedder
from copilot.embeddings.fake import HashEmbedder
from copilot.embeddings.fastembed_embedder import FastEmbedEmbedder
from copilot.embeddings.representation import REPRESENTATION_VERSION, embedding_text

__all__ = [
    "REPRESENTATION_VERSION",
    "ChunkEmbeddings",
    "Embedder",
    "EmbeddingConfigError",
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingModelUnavailableError",
    "EmbeddingRuntimeError",
    "FastEmbedEmbedder",
    "HashEmbedder",
    "TokenCountingEmbedder",
    "create_embedder",
    "embed_chunks",
    "embedding_text",
]
