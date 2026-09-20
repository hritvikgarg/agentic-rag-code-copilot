"""Embedding errors. Wrapped errors always keep the original exception as ``__cause__``."""

from __future__ import annotations


class EmbeddingError(Exception):
    """Base class for embedding failures."""


class EmbeddingConfigError(EmbeddingError):
    """Invalid configuration: unknown/unsupported model id, fastembed not installed, ..."""


class EmbeddingModelUnavailableError(EmbeddingError):
    """The model could not be downloaded or loaded (network, cache, corrupt files, ...)."""


class EmbeddingRuntimeError(EmbeddingError):
    """The model ran but produced unusable output (wrong shape, non-finite values, ...)."""


class EmbeddingInputError(EmbeddingError, ValueError):
    """The caller passed invalid text (wrong type, blank string, bare ``str`` instead of a list)."""
