"""Input validation and output post-processing shared by every embedder.

**Normalisation decision.** The embedding layer guarantees *unit-length (L2-normalised) float32
vectors*. Reason: cosine similarity is ``dot(a, b) / (|a||b|)``, which equals a plain dot product
when both vectors have length 1. Normalising once, here, means the vector index (FAISS
``IndexFlatIP`` in Milestone 5) needs no extra step and cannot silently mix normalised and
un-normalised vectors. The step is idempotent, so a model that already returns unit vectors
(the Jina code model does, via fastembed) is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from copilot.embeddings.errors import EmbeddingInputError, EmbeddingRuntimeError

Vectors = np.ndarray  # float32, shape (n, dim)


def validate_texts(texts: Sequence[str]) -> list[str]:
    """Validate a batch of texts to embed and return it as a list.

    Defined behaviour: an empty sequence is valid (yields zero vectors); a bare ``str`` is
    rejected (it would silently be embedded character by character); every item must be a
    non-blank ``str``.
    """
    if isinstance(texts, str):
        raise EmbeddingInputError(
            "expected a sequence of texts, got a single str; wrap it in a list"
        )
    try:
        items = list(texts)
    except TypeError as exc:
        raise EmbeddingInputError(
            f"expected a sequence of texts, got {type(texts).__name__}"
        ) from exc
    for i, item in enumerate(items):
        validate_text(item, where=f"texts[{i}]")
    return items


def validate_text(text: object, *, where: str = "text") -> str:
    """Validate one text: it must be a ``str`` with at least one non-whitespace character."""
    if not isinstance(text, str):
        raise EmbeddingInputError(f"{where} must be str, got {type(text).__name__}")
    if not text.strip():
        raise EmbeddingInputError(f"{where} is empty or whitespace-only")
    return text


def finalize_vectors(raw: object, *, count: int, dimension: int) -> Vectors:
    """Check shape and finiteness of model output, L2-normalise, and return float32 ``(n, dim)``."""
    try:
        array = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EmbeddingRuntimeError("model output could not be converted to a float array") from exc
    if count == 0:
        return np.empty((0, dimension), dtype=np.float32)
    if array.shape != (count, dimension):
        raise EmbeddingRuntimeError(
            f"model returned shape {array.shape}, expected ({count}, {dimension})"
        )
    if not np.isfinite(array).all():
        raise EmbeddingRuntimeError("model returned non-finite values (NaN or inf)")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms == 0).any():
        raise EmbeddingRuntimeError("model returned an all-zero vector; cannot normalise")
    out = (array / norms).astype(np.float32, copy=False)
    out.setflags(write=False)
    return out
