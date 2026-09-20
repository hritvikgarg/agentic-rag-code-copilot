"""Chunking strategies: baseline line windows (Milestone 3); structure-aware AST chunks (M9)."""

from copilot.chunking.base import Chunker, chunk_repository
from copilot.chunking.errors import (
    ChunkingError,
    StrategyNotImplementedError,
    UnknownChunkingStrategyError,
)
from copilot.chunking.line_chunker import LineChunker
from copilot.chunking.registry import available_strategies, create_chunker

__all__ = [
    "ChunkingError",
    "Chunker",
    "LineChunker",
    "StrategyNotImplementedError",
    "UnknownChunkingStrategyError",
    "available_strategies",
    "chunk_repository",
    "create_chunker",
]
