"""Deterministic hashed bag-of-words embedder for tests and offline development.

**Not a real embedding model.** It hashes each word into one of ``dimension`` buckets with a
pseudo-random sign and L2-normalises, so texts sharing words have higher cosine similarity. It
exists so the pipeline (chunk pairing, index, retrieval plumbing) can be tested without a 0.6 GB
model download. Quality claims must never be made from it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from copilot.embeddings.vectors import finalize_vectors, validate_text, validate_texts
from copilot.models.embedding import EmbeddingModelInfo
from copilot.utils.tokens import estimate_tokens

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class HashEmbedder:
    """Fake embedder: deterministic across runs and platforms (uses BLAKE2b, not ``hash()``)."""

    def __init__(self, dimension: int = 64, *, batch_size: int = 32) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self._info = EmbeddingModelInfo(
            model_id="fake/hash-embedder",
            dimension=dimension,
            runtime="python-hash",
            normalized=True,
            pooling="bag-of-words",
            max_input_tokens=None,
            batch_size=batch_size,
        )

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def _embed_one(self, text: str) -> np.ndarray:
        dim = self._info.dimension
        vec = np.zeros(dim, dtype=np.float64)
        for word in _WORD_RE.findall(text.lower()) or [text]:
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vec[value % dim] += 1.0 if (value >> 32) & 1 else -1.0
        if not vec.any():  # opposite signs cancelled exactly; fall back to the whole text
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            vec[int.from_bytes(digest, "big") % dim] = 1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        items = validate_texts(texts)
        raw = [self._embed_one(t) for t in items]
        return finalize_vectors(raw, count=len(items), dimension=self._info.dimension)

    def embed_query(self, text: str) -> np.ndarray:
        validate_text(text)
        return finalize_vectors([self._embed_one(text)], count=1, dimension=self._info.dimension)[0]

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Uses the project's heuristic estimator (there is no real tokenizer here)."""
        return [estimate_tokens(t) for t in validate_texts(texts)]
