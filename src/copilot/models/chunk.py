"""Chunk schema shared by every chunking strategy (Milestone 3 onwards).

A *chunk* is the unit that is embedded, indexed, retrieved and cited. The schema is strategy
neutral: the baseline line-window strategy leaves the symbol fields ``None``; the structure-aware
strategy (Milestone 9) will fill them. Keeping one schema is what lets the two strategies be
compared with identical downstream code.

Invariants (enforced by the model or the chunkers, and covered by tests):

* ``1 <= start_line <= end_line`` (1-based, inclusive, lines of the *normalised* source text).
* ``content`` equals the source lines ``start_line..end_line`` joined with ``"\\n"`` **unless**
  ``line_fragment`` is true, in which case ``content`` is a piece of one over-long line and
  ``start_line == end_line``.
* ``chunk_id`` is a deterministic function of the chunk's identity, so re-chunking the same
  repository yields the same ids.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from copilot.utils.paths import validate_relative_posix


class ChunkType(StrEnum):
    """What kind of text a chunk holds. Structure-aware types are added in Milestone 9."""

    CODE_WINDOW = "code_window"  # fixed-size line window over source code
    CONFIG_WINDOW = "config_window"  # fixed-size line window over JSON/YAML
    DOC_SECTION = "doc_section"  # Markdown section (or a window of an oversized section)


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_chunk_id(
    *,
    repository_name: str,
    chunking_strategy: str,
    file_path: str,
    chunk_index: int,
    start_line: int,
    end_line: int,
    content_sha256: str,
) -> str:
    """Deterministic 16-hex-character id for a chunk.

    Includes ``chunk_index`` so two chunks of one file with identical text (e.g. repeated
    fragments of a long line) can never collide.
    """
    key = "\x00".join(
        [
            repository_name,
            chunking_strategy,
            file_path,
            str(chunk_index),
            str(start_line),
            str(end_line),
            content_sha256,
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class Chunk(BaseModel):
    """One retrievable piece of a repository file, with citation metadata."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    repository_name: str = Field(min_length=1)
    file_path: str  # canonical repository-relative POSIX path
    language: str
    chunk_type: ChunkType
    chunking_strategy: str = Field(min_length=1)  # registry name, e.g. "line"
    chunk_index: int = Field(ge=0)  # 0-based position of this chunk within its file
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str = Field(repr=False)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_estimate: int = Field(ge=0)
    line_fragment: bool = False  # True: content is part of a single over-long line

    # Symbol metadata. Always None for the baseline strategy; reserved for Milestone 9.
    symbol_name: str | None = None
    qualified_name: str | None = None
    parent_class: str | None = None
    # Markdown heading path such as "Guide > Install" (doc sections only).
    heading: str | None = None

    @field_validator("file_path")
    @classmethod
    def _canonical_file_path(cls, value: str) -> str:
        return validate_relative_posix(value)

    @model_validator(mode="after")
    def _check_consistency(self) -> Chunk:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if self.line_fragment and self.start_line != self.end_line:
            raise ValueError("a line fragment must lie on a single line")
        if sha256_text(self.content) != self.content_sha256:
            raise ValueError("content_sha256 does not match content")
        return self


class ChunkStats(BaseModel):
    """Summary statistics over a set of chunks (feeds the chunking experiment)."""

    model_config = ConfigDict(frozen=True)

    strategy: str
    files_chunked: int  # files that produced at least one chunk
    files_without_chunks: int  # e.g. empty or whitespace-only files
    chunk_count: int
    line_fragment_count: int
    tokens_mean: float
    tokens_median: float
    tokens_p95: float
    tokens_max: int
    by_chunk_type: dict[str, int]
    by_language: dict[str, int]
    elapsed_seconds: float

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        return (
            f"strategy={self.strategy} chunks={self.chunk_count} "
            f"files_chunked={self.files_chunked} files_without_chunks={self.files_without_chunks} "
            f"tokens(mean/median/p95/max)="
            f"{self.tokens_mean:.1f}/{self.tokens_median:.1f}/{self.tokens_p95:.1f}/"
            f"{self.tokens_max} line_fragments={self.line_fragment_count} "
            f"elapsed={self.elapsed_seconds:.3f}s"
        )


class ChunkingResult(BaseModel):
    """All chunks for a repository (ordered by file path, then position) plus statistics."""

    model_config = ConfigDict(frozen=True)

    repository_name: str
    chunks: tuple[Chunk, ...]
    stats: ChunkStats
