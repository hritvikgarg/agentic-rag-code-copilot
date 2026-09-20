"""Shared data schemas (ingestion and chunk models now; answer schemas in later milestones)."""

from copilot.models.chunk import Chunk, ChunkingResult, ChunkStats, ChunkType
from copilot.models.ingestion import (
    IngestionResult,
    IngestionStats,
    SkippedFile,
    SkipReason,
    SourceFile,
)

__all__ = [
    "Chunk",
    "ChunkStats",
    "ChunkType",
    "ChunkingResult",
    "IngestionResult",
    "IngestionStats",
    "SkipReason",
    "SkippedFile",
    "SourceFile",
]
