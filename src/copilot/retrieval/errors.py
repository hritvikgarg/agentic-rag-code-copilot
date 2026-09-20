"""Retrieval errors: every failure mode has its own application-specific type.

Nothing is repaired or guessed. In particular a stale or altered repository never yields stale
source text: it raises a ``MaterializationError`` subclass.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Base class for all retrieval failures."""


class QueryError(RetrievalError, ValueError):
    """The query text or ``top_k`` is invalid (wrong type, empty, blank, too long, out of range)."""


class EmbedderMismatchError(RetrievalError):
    """The query embedder is not the one the index was built with (model id, dimension, norm)."""


class MaterializationError(RetrievalError):
    """Source text for a retrieved chunk could not be reproduced faithfully."""


class StaleRepositoryError(MaterializationError):
    """The repository no longer matches the index (its fingerprint differs).

    A file was added, removed or edited, or a different ``--ignore-dir`` set / repository name is
    being used than at build time. Rebuild the index, or point at the tree that was indexed.
    """


class ChunkNotFoundError(MaterializationError):
    """A chunk listed in the index cannot be found when the repository is re-chunked."""


class ChunkContentMismatchError(MaterializationError):
    """A re-chunked chunk has the recorded id but different content, location or source hash."""
