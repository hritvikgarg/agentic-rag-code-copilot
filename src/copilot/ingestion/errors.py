"""Exceptions raised by repository ingestion."""


class IngestionError(Exception):
    """Base class for ingestion failures that abort a run."""


class InvalidRepositoryError(IngestionError):
    """The repository root is missing, not a directory, or unsafe to scan (e.g. filesystem root)."""
