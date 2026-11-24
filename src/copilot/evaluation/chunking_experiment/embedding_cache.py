"""Resumable document embedding: the same vectors as one continuous run, in time-boxed slices.

``ResumableEmbedder`` wraps a real embedder. ``embed_documents`` processes the texts in the
embedder's own batch size, in input order (the same batches a single uninterrupted call would
form) and stores every finished batch (vectors + measured seconds) in a cache directory keyed by the
hash of the batch's texts. When a time budget is exhausted it raises ``TimeBudgetExceeded``; the
caller simply runs the same command again and finished batches are loaded instead of recomputed.

Why this exists: some environments limit how long one command may run (the Linux sandbox used for
Milestone 10 allows about three minutes per call) while embedding a few hundred 500-token chunks
takes several minutes. It is **optional** (``--time-budget``); without it the build embeds in one
go. The sum of the stored per-batch seconds is the arm's measured embedding time; model loading is
reported separately and is never part of it.

One cache directory belongs to one arm, so no arm profits from another arm's work (each arm's
measured embedding cost is its own).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from copilot.embeddings.base import Embedder
from copilot.models.embedding import EmbeddingModelInfo


def _now() -> float:
    """The clock used for the time budget and the per-batch timing (replaceable in tests)."""
    return time.perf_counter()


class TimeBudgetExceeded(RuntimeError):
    """The embedding time budget ran out; run the same command again to continue."""

    def __init__(self, done_batches: int, total_batches: int) -> None:
        super().__init__(f"embedding paused after {done_batches}/{total_batches} batches")
        self.done_batches = done_batches
        self.total_batches = total_batches


def _batch_key(model_id: str, number: int, texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([model_id, number, len(texts)]).encode())
    for text in texts:
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
    return digest.hexdigest()[:24]


class ResumableEmbedder:
    """An ``Embedder`` whose document embedding survives interruption."""

    def __init__(
        self,
        inner: Embedder,
        cache_dir: Path,
        *,
        batch_size: int,
        time_budget_seconds: float | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._inner = inner
        self._dir = cache_dir
        self._batch_size = batch_size
        self._budget = time_budget_seconds
        self._started: float | None = None
        self.recorded_seconds = 0.0  # measured seconds of every batch used by this build
        self.batches_computed = 0  # computed in this process
        self.batches_loaded = 0  # loaded from the cache

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._inner.info

    def embed_query(self, text: str) -> np.ndarray:
        return self._inner.embed_query(text)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._started is None:
            self._started = _now()
        model_id = self.info.model_id
        total = -(-len(texts) // self._batch_size)
        parts: list[np.ndarray] = []
        for number in range(total):
            batch = list(texts[number * self._batch_size : (number + 1) * self._batch_size])
            path = self._dir / f"{number:05d}-{_batch_key(model_id, number, batch)}.npz"
            if path.is_file():
                try:
                    with np.load(path) as stored:
                        vectors, seconds = stored["vectors"], float(stored["seconds"])
                    if vectors.shape[0] == len(batch):
                        parts.append(vectors)
                        self.recorded_seconds += seconds
                        self.batches_loaded += 1
                        continue
                except (OSError, ValueError, KeyError):
                    pass  # unreadable cache entry: recompute below
            if self._budget is not None and _now() - self._started > self._budget:
                raise TimeBudgetExceeded(number, total)
            began = _now()
            vectors = self._inner.embed_documents(batch)
            seconds = _now() - began
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, vectors=vectors, seconds=np.float64(seconds))
            tmp.replace(path)
            parts.append(vectors)
            self.recorded_seconds += seconds
            self.batches_computed += 1
        if not parts:
            return self._inner.embed_documents([])
        return np.vstack(parts)
