"""Pairing chunk ids with vectors: the hand-off to the vector store (Milestone 5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from copilot.embeddings.base import Embedder
from copilot.embeddings.representation import REPRESENTATION_VERSION, TextStyle, embedding_text
from copilot.models.chunk import Chunk
from copilot.models.embedding import EmbeddingModelInfo


@dataclass(frozen=True)
class ChunkEmbeddings:
    """Vectors for a list of chunks: ``vectors[i]`` belongs to ``chunk_ids[i]``.

    Holds ids and vectors only (no chunk text), plus the provenance an index manifest needs.
    """

    chunk_ids: tuple[str, ...]
    vectors: np.ndarray  # float32, shape (n, dimension), unit length, read-only
    model: EmbeddingModelInfo
    text_style: TextStyle
    representation_version: int

    def __post_init__(self) -> None:
        if self.vectors.shape != (len(self.chunk_ids), self.model.dimension):
            raise ValueError(
                f"vectors shape {self.vectors.shape} does not match "
                f"({len(self.chunk_ids)}, {self.model.dimension})"
            )
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids must be unique")


def embed_chunks(
    embedder: Embedder, chunks: Sequence[Chunk], *, style: TextStyle = "prefixed"
) -> ChunkEmbeddings:
    """Embed ``chunks`` (as documents) using the given text representation."""
    texts = [embedding_text(c, style) for c in chunks]
    vectors = embedder.embed_documents(texts)
    return ChunkEmbeddings(
        chunk_ids=tuple(c.chunk_id for c in chunks),
        vectors=vectors,
        model=embedder.info,
        text_style=style,
        representation_version=REPRESENTATION_VERSION,
    )
