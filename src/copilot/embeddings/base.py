"""The embedding interface the rest of the application depends on.

Nothing outside ``copilot.embeddings`` imports fastembed; replacing the model or runtime means
writing another class with these methods.

**Query vs document behaviour.** Some embedding models are *asymmetric* (queries need a prefix or a
different head than documents). The Jina code model is not: fastembed documents "prefixes for
queries/documents: not necessary". Both methods still exist so callers never need to know, and so
an asymmetric model can be swapped in without touching call sites. Both produce vectors in the
same space with the same dimension.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from copilot.models.embedding import EmbeddingModelInfo


@runtime_checkable
class Embedder(Protocol):
    """Turns text into unit-length float32 vectors."""

    @property
    def info(self) -> EmbeddingModelInfo:
        """Description of the model (may load it on first access)."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts to embed as documents. Returns ``(len(texts), dimension)``, in input order.

        An empty sequence returns shape ``(0, dimension)``. Raises ``EmbeddingInputError`` for a
        bare ``str``, a non-``str`` item, or an empty/whitespace-only item.
        """
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one search query. Returns shape ``(dimension,)`` in the documents' space."""
        ...


@runtime_checkable
class TokenCountingEmbedder(Embedder, Protocol):
    """An embedder that can also report how many model tokens a text occupies."""

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Token count per text **without truncation**, as the model's tokenizer sees it."""
        ...
