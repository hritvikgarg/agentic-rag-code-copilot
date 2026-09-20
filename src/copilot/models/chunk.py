"""Chunk schema shared by every chunking strategy (Milestone 3 onwards).

A *chunk* is the unit that is embedded, indexed, retrieved and cited. The schema is strategy
neutral: the baseline line-window strategy leaves the symbol fields ``None``; the structure-aware
strategy (Milestone 9) will fill them and add its own chunk types.

Invariants (enforced by the model or the chunkers, and covered by tests):

* ``1 <= start_line <= end_line`` (1-based, inclusive, lines of the *normalised* source text; a
  terminal newline does not create an extra line).
* For a chunk of whole lines (``fragment_index is None``) ``content`` equals the source lines
  ``start_line..end_line`` joined with ``"\\n"``.
* For a fragment of an over-long physical line, ``start_line == end_line`` is that physical
  line's own number, ``fragment_index``/``fragment_count`` identify the piece, and ``content`` is
  that piece of the line. All fragments of a line concatenate to the exact line.
* ``chunk_id`` follows the contract documented on :func:`make_chunk_id` and in
  ``docs/chunking.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from copilot.utils.paths import validate_relative_posix

CHUNK_ID_SCHEMA = "chunk-id/1"  # bump if the id formula or serialisation ever changes


class ChunkType(StrEnum):
    """What kind of unit a chunk is. Structure-aware types are added in Milestone 9."""

    LINE_WINDOW = "line_window"  # window of whole lines (or a fragment of one over-long line)


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_chunk_id(
    *,
    repository_name: str,
    file_path: str,
    source_sha256: str,
    strategy: str,
    strategy_version: int,
    params: Mapping[str, int | str],
    start_line: int,
    end_line: int,
    fragment_index: int | None,
) -> str:
    """Deterministic chunk id: first 16 hex characters of SHA-256 over a canonical JSON list.

    The hashed text is ``json.dumps(payload, ensure_ascii=True, sort_keys=True,
    separators=(",", ":"))`` where ``payload`` is::

        [CHUNK_ID_SCHEMA, repository_name, file_path, source_sha256,
         strategy, strategy_version, params, unit]

    and ``unit`` is ``["w", start_line, end_line]`` for a window of whole lines or
    ``["f", line, fragment_index]`` for a fragment of one physical line. JSON quoting makes the
    serialisation unambiguous (no delimiter can be forged by a path or name). ``params`` is the
    chunker's parameter mapping (sorted by key via ``sort_keys``).

    What changes an id: the repository name, the path, the source version (``source_sha256`` is
    the SHA-256 of the file's raw bytes, so **any** edit to a file changes every chunk id of that
    file), the strategy name, the strategy algorithm version, any chunker parameter, and the
    chunk's position (line range, or line + fragment index). Nothing else, and nothing random.
    """
    unit: list[int | str] = (
        ["w", start_line, end_line] if fragment_index is None else ["f", start_line, fragment_index]
    )
    payload = [
        CHUNK_ID_SCHEMA,
        repository_name,
        file_path,
        source_sha256,
        strategy,
        strategy_version,
        dict(params),
        unit,
    ]
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Chunk(BaseModel):
    """One retrievable piece of a repository file, with citation metadata."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    repository_name: str = Field(min_length=1)
    file_path: str  # canonical repository-relative POSIX path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # SourceFile.sha256 (raw-bytes hash)
    language: str
    chunk_type: ChunkType
    chunking_strategy: str = Field(min_length=1)  # registry name, e.g. "line"
    chunking_version: int = Field(ge=1)  # algorithm version of the strategy
    chunk_index: int = Field(ge=0)  # 0-based position of this chunk within its file
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str = Field(repr=False)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # SHA-256 of ``content``
    token_estimate: int = Field(ge=0)

    # Set only for a fragment of an over-long physical line (see module docstring).
    fragment_index: int | None = Field(default=None, ge=0)
    fragment_count: int | None = Field(default=None, ge=1)

    # Symbol metadata. Always None for the baseline strategy; reserved for Milestone 9.
    symbol_name: str | None = None
    qualified_name: str | None = None
    parent_class: str | None = None

    @field_validator("file_path")
    @classmethod
    def _canonical_file_path(cls, value: str) -> str:
        return validate_relative_posix(value)

    @property
    def is_fragment(self) -> bool:
        """True if this chunk is a piece of a single over-long physical line."""
        return self.fragment_index is not None

    @model_validator(mode="after")
    def _check_consistency(self) -> Chunk:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if (self.fragment_index is None) != (self.fragment_count is None):
            raise ValueError("fragment_index and fragment_count must be set together")
        if self.fragment_index is not None:
            assert self.fragment_count is not None
            if self.fragment_index >= self.fragment_count:
                raise ValueError("fragment_index must be < fragment_count")
            if self.start_line != self.end_line:
                raise ValueError("a fragment must lie on a single physical line")
        if sha256_text(self.content) != self.content_sha256:
            raise ValueError("content_sha256 does not match content")
        return self


class ChunkStats(BaseModel):
    """Summary statistics over a set of chunks (feeds the chunking experiment).

    Line accounting distinguishes *physical source lines* from *chunk line slots*:

    * ``source_lines_total``: physical lines of all files given to the chunker (a terminal
      newline is not a line).
    * ``unique_source_lines_represented``: distinct (file, line) pairs covered by at least one
      chunk. A fragmented line counts once, however many fragments it has. Whitespace-only lines
      that fall in dropped windows are not represented.
    * ``overlap_duplicated_lines``: extra copies of lines introduced by overlap, i.e. the sum of
      line counts of whole-line chunks minus the distinct lines they cover. Fragments never
      contribute (they add no physical lines).
    * ``chunk_count = whole_line_chunk_count + fragment_chunk_count``.
    * ``fragmented_line_count``: distinct physical lines that had to be fragmented.
    """

    model_config = ConfigDict(frozen=True)

    strategy: str
    files_chunked: int  # files that produced at least one chunk
    files_without_chunks: int  # e.g. empty or whitespace-only files
    chunk_count: int
    whole_line_chunk_count: int
    fragment_chunk_count: int
    fragmented_line_count: int
    source_lines_total: int
    unique_source_lines_represented: int
    overlap_duplicated_lines: int
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
            f"(whole_line={self.whole_line_chunk_count} fragments={self.fragment_chunk_count}) "
            f"files_chunked={self.files_chunked} files_without_chunks={self.files_without_chunks} "
            f"lines(total/represented/overlap_duplicates)="
            f"{self.source_lines_total}/{self.unique_source_lines_represented}/"
            f"{self.overlap_duplicated_lines} fragmented_lines={self.fragmented_line_count} "
            f"tokens(mean/median/p95/max)="
            f"{self.tokens_mean:.1f}/{self.tokens_median:.1f}/{self.tokens_p95:.1f}/"
            f"{self.tokens_max} elapsed={self.elapsed_seconds:.3f}s"
        )


class ChunkingResult(BaseModel):
    """All chunks for a repository (ordered by file path, then position) plus statistics."""

    model_config = ConfigDict(frozen=True)

    repository_name: str
    chunks: tuple[Chunk, ...]
    stats: ChunkStats
