"""Typed models produced by repository ingestion (Milestone 2).

Design notes:
* **No absolute host paths.** Models carry only the repository name and a repository-relative
  POSIX path. Absolute paths differ between machines (Windows/Linux/deployment), would leak the
  user's directory layout into indexes and logs, and are never needed downstream: a consumer that
  needs the file re-derives it with ``safe_join(root, relative_path)``.
* **No content in ``repr``.** ``SourceFile.content`` is excluded from ``repr`` so accidentally
  logging an object cannot dump file contents.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from copilot.utils.paths import validate_relative_posix


class SkipReason(StrEnum):
    """Why a file was not ingested. Values are stable strings used as statistics keys."""

    UNSUPPORTED_EXTENSION = "unsupported_extension"
    IGNORED_FILE = "ignored_file"  # generated/vendored files such as lock files, *.min.js
    SENSITIVE = "sensitive"  # names that indicate secrets: .env, *.pem, credentials.json, ...
    BINARY = "binary"
    OVERSIZED = "oversized"
    UNREADABLE = "unreadable"  # OS-level read error (permissions, file vanished, ...)
    ENCODING_ERROR = "encoding_error"  # not valid UTF-8 (or UTF-16 with BOM)
    UNSAFE_PATH = "unsafe_path"  # symlink target outside the root, or an unsafe file name
    SYMLINK = "symlink"  # symlinks are never followed (policy), target inside the root
    SPECIAL_FILE = "special_file"  # FIFO, socket, device, ...
    LIMIT_EXCEEDED = "limit_exceeded"  # repository-wide file-count or total-size limit hit


class SourceFile(BaseModel):
    """One accepted repository file, ready for chunking."""

    model_config = ConfigDict(frozen=True)

    repository_name: str = Field(min_length=1)
    relative_path: str  # canonical repository-relative POSIX path, e.g. "src/app/main.py"
    extension: str  # lower-case with leading dot, e.g. ".py"
    language: str  # from the central extension mapping, e.g. "python"
    size_bytes: int = Field(ge=0)  # size of the raw file on disk
    content: str = Field(repr=False)  # decoded text, newlines normalised to "\n"
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # SHA-256 of the raw bytes (change detection)

    @field_validator("relative_path")
    @classmethod
    def _canonical_relative_path(cls, value: str) -> str:
        return validate_relative_posix(value)


class SkippedFile(BaseModel):
    """A file (or symlink) that was seen but not ingested. Never carries content."""

    model_config = ConfigDict(frozen=True)

    relative_path: str
    reason: SkipReason


class IngestionStats(BaseModel):
    """Counters describing one ingestion run."""

    model_config = ConfigDict(frozen=True)

    # "Discovered" means non-directory entries encountered in directories that were traversed.
    # Files inside pruned directories are never enumerated, so they are not counted here.
    files_discovered: int
    files_accepted: int
    files_skipped: int
    skip_reasons: dict[str, int]  # SkipReason value -> count
    directories_pruned: int
    pruned_directory_names: dict[str, int]  # directory name -> times pruned
    directories_unreadable: int
    total_bytes_accepted: int
    languages: dict[str, int]  # language -> accepted file count
    truncated: bool  # True if a repository-wide limit stopped the scan early
    truncation_reason: str | None
    elapsed_seconds: float

    def summary(self) -> str:
        """One-paragraph human-readable summary (no paths, no contents)."""
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(self.skip_reasons.items())) or "none"
        text = (
            f"discovered={self.files_discovered} accepted={self.files_accepted} "
            f"skipped={self.files_skipped} ({reasons}); "
            f"directories_pruned={self.directories_pruned}; "
            f"bytes_accepted={self.total_bytes_accepted}; elapsed={self.elapsed_seconds:.3f}s"
        )
        if self.truncated:
            text += f"; TRUNCATED ({self.truncation_reason})"
        return text


class IngestionResult(BaseModel):
    """Everything ingestion returns: accepted files, skipped files and statistics."""

    model_config = ConfigDict(frozen=True)

    repository_name: str
    files: tuple[SourceFile, ...]  # sorted by relative_path
    skipped: tuple[SkippedFile, ...]  # sorted by relative_path
    stats: IngestionStats

    @property
    def relative_paths(self) -> list[str]:
        """Relative paths of accepted files, in canonical (sorted) order."""
        return [f.relative_path for f in self.files]
