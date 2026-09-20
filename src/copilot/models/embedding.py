"""Typed description of an embedding model (Milestone 4).

Deliberately contains **no vectors and no chunk content**: it describes *which* model produced a
set of vectors so an index can record it (and refuse to mix incompatible vectors). Vectors are
paired with chunks by ``chunk_id`` in ``copilot.embeddings.chunk_embeddings.ChunkEmbeddings``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingModelInfo(BaseModel):
    """What an embedder is, and what its vectors look like."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(min_length=1)  # e.g. "jinaai/jina-embeddings-v2-base-code"
    dimension: int = Field(ge=1)  # length of every vector
    runtime: str  # e.g. "fastembed-onnx"
    runtime_version: str | None = None  # library version, when known
    normalized: bool  # True: every vector has unit L2 norm (cosine == dot product)
    pooling: str | None = None  # how token vectors are combined, e.g. "mean"
    max_input_tokens: int | None = None  # tokenizer truncation limit (None when unknown)
    batch_size: int = Field(ge=1)  # texts per forward pass
    asymmetric: bool = False  # True if queries and documents need different handling
