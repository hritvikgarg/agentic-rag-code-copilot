"""Safe repository discovery and loading (Milestone 2)."""

from copilot.ingestion.errors import IngestionError, InvalidRepositoryError
from copilot.ingestion.languages import DEFAULT_LANGUAGE_BY_EXTENSION, detect_language
from copilot.ingestion.loader import ingest_repository, resolve_repository_root
from copilot.ingestion.policy import IngestionPolicy

__all__ = [
    "DEFAULT_LANGUAGE_BY_EXTENSION",
    "IngestionError",
    "IngestionPolicy",
    "InvalidRepositoryError",
    "detect_language",
    "ingest_repository",
    "resolve_repository_root",
]
