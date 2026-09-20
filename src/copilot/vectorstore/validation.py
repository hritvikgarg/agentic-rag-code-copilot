"""Pre-indexing consistency checks. Every failure raises ``IndexBuildError`` (fail loudly).

What is verified before anything is stored:

* the vectors are a 2-D ``float32`` array with the expected dimension;
* every value is finite (no NaN/inf);
* every vector has unit length within ``NORM_TOLERANCE`` (the index computes cosine similarity as a
  plain inner product, which is only correct for unit vectors);
* chunk ids are unique, and there is exactly one vector per chunk;
* chunks are in the canonical order (see ``canonical_order_key``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from copilot.models.chunk import Chunk
from copilot.vectorstore.errors import EmptyIndexError, IndexBuildError

# Vectors are float32; a unit vector's norm is 1 +- ~1e-6 in practice. 1e-3 is loose enough to never
# reject a healthy model and tight enough to catch an un-normalised vector (norm typically >> 1).
NORM_TOLERANCE = 1e-3


def canonical_order_key(chunk: Chunk) -> tuple[str, int]:
    """Sort key of the vector-ordering contract: ``(file_path, chunk_index)``.

    ``file_path`` is compared as a Python string (Unicode code points), which is identical on every
    platform, and ``chunk_index`` numerically. The pair is unique per chunk (``chunk_index`` counts
    chunks within a file), so the order is total and independent of directory-traversal order.
    """
    return (chunk.file_path, chunk.chunk_index)


def sort_chunks_canonically(chunks: Sequence[Chunk]) -> list[Chunk]:
    """``chunks`` in canonical order (a new list)."""
    return sorted(chunks, key=canonical_order_key)


def validate_vectors(vectors: object, *, dimension: int) -> np.ndarray:
    """Check shape, dtype, finiteness and unit length; return the array unchanged."""
    if not isinstance(vectors, np.ndarray):
        raise IndexBuildError(f"vectors must be a numpy array, got {type(vectors).__name__}")
    if vectors.dtype != np.float32:
        raise IndexBuildError(f"vectors must be float32, got {vectors.dtype}")
    if vectors.ndim != 2:
        raise IndexBuildError(f"vectors must be 2-D (n, dimension), got shape {vectors.shape}")
    if vectors.shape[0] == 0:
        raise EmptyIndexError("no vectors to index")
    if vectors.shape[1] != dimension:
        raise IndexBuildError(
            f"vector dimension {vectors.shape[1]} does not match the expected {dimension}"
        )
    if not np.isfinite(vectors).all():
        raise IndexBuildError("vectors contain NaN or infinite values")
    deviation = float(np.abs(np.linalg.norm(vectors, axis=1) - 1.0).max())
    if deviation > NORM_TOLERANCE:
        raise IndexBuildError(
            f"vectors are not unit length (max |norm - 1| = {deviation:.3g}, "
            f"tolerance {NORM_TOLERANCE}); cosine similarity would be wrong"
        )
    return vectors


def validate_chunks(chunks: Sequence[Chunk], *, vector_count: int) -> None:
    """Check emptiness, count agreement, id uniqueness and canonical order."""
    if not chunks:
        raise EmptyIndexError("no chunks to index")
    if len(chunks) != vector_count:
        raise IndexBuildError(f"{len(chunks)} chunks but {vector_count} vectors")
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise IndexBuildError("duplicate chunk ids")
    keys = [canonical_order_key(c) for c in chunks]
    if keys != sorted(keys):
        raise IndexBuildError(
            "chunks are not in canonical order (file_path, chunk_index); "
            "use sort_chunks_canonically() and embed in that order"
        )
