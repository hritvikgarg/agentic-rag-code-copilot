"""Shared data schemas (ingestion models now; chunk and answer schemas in later milestones)."""

from copilot.models.ingestion import (
    IngestionResult,
    IngestionStats,
    SkippedFile,
    SkipReason,
    SourceFile,
)

__all__ = ["IngestionResult", "IngestionStats", "SkipReason", "SkippedFile", "SourceFile"]
