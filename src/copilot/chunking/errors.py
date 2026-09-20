"""Chunking errors."""

from __future__ import annotations


class ChunkingError(Exception):
    """Base class for chunking failures."""


class UnknownChunkingStrategyError(ChunkingError, ValueError):
    """The requested strategy name is not known at all."""


class StrategyNotImplementedError(ChunkingError):
    """The strategy name is reserved for a later milestone but has no implementation yet."""
