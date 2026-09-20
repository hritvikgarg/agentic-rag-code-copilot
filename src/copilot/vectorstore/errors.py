"""Vector-store errors: every failure mode has its own application-specific type.

Callers can catch ``VectorStoreError`` for "anything went wrong with the index", or a subclass to
react precisely (e.g. rebuild on ``IndexCompatibilityError``). Loading never rebuilds silently.
"""

from __future__ import annotations


class VectorStoreError(Exception):
    """Base class for all vector-store failures."""


class IndexBuildError(VectorStoreError, ValueError):
    """The inputs to ``build`` are inconsistent (count, dimension, NaN, norm, duplicates...)."""


class EmptyIndexError(IndexBuildError):
    """There is nothing to index (no accepted files, zero chunks or zero vectors)."""


class SearchInputError(VectorStoreError, ValueError):
    """A search request is malformed (query-vector shape/dtype/norm/finiteness, or ``top_k``)."""


class IndexStorageError(VectorStoreError):
    """Writing the index failed (I/O error, invalid target)."""


class IndexExistsError(IndexStorageError):
    """An index with this id is already stored; pass ``overwrite=True`` to replace it."""


class IndexCorruptError(VectorStoreError):
    """A stored index is damaged or internally inconsistent.

    Examples: a missing artifact, unparsable manifest, size/SHA-256 mismatch, a FAISS ``ntotal``
    that disagrees with the manifest, or a chunk mapping with the wrong length.
    """


class IndexCompatibilityError(VectorStoreError):
    """A stored index is intact but does not match what the caller expects.

    Raised for an unsupported manifest schema, or when the manifest differs from the expected
    embedding model/dimension/representation/chunking configuration/repository fingerprint/index
    type. ``mismatches`` lists ``(field, expected, actual)`` for every difference found.
    """

    def __init__(self, message: str, mismatches: tuple[tuple[str, object, object], ...] = ()):
        super().__init__(message)
        self.mismatches = mismatches
