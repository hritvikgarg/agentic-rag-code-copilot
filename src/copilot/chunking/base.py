"""The chunker interface and shared helpers.

Every strategy (baseline line windows now, AST structure-aware in Milestone 9) implements the
same tiny ``Chunker`` protocol, so the indexer, retriever and evaluation harness never need to
know which strategy produced a chunk. That is what makes the two-strategy experiment fair.
"""

from __future__ import annotations

import time
from typing import Protocol

from copilot.chunking.errors import ChunkingError
from copilot.chunking.stats import compute_stats
from copilot.chunking.windows import Window, split_lines
from copilot.models.chunk import Chunk, ChunkingResult, ChunkType, make_chunk_id, sha256_text
from copilot.models.ingestion import IngestionResult, SourceFile
from copilot.utils.tokens import estimate_tokens


class Chunker(Protocol):
    """Turns one ingested file into chunks."""

    @property
    def name(self) -> str:
        """Registry name of the strategy, e.g. ``"line"``. Stored on every chunk."""
        ...

    @property
    def version(self) -> int:
        """Algorithm version; bump whenever output for the same input/params could change."""
        ...

    def params(self) -> dict[str, int | str]:
        """Parameters that affect output; part of every chunk id and the index manifest (M4)."""
        ...

    def chunk_file(self, file: SourceFile) -> list[Chunk]:
        """Chunks of ``file`` in file order (deterministic)."""
        ...


def build_chunk(
    file: SourceFile,
    chunker: Chunker,
    *,
    index: int,
    window: Window,
    chunk_type: ChunkType,
) -> Chunk:
    """Create a validated ``Chunk`` from a window of ``file`` (id per ``make_chunk_id``)."""
    return Chunk(
        chunk_id=make_chunk_id(
            repository_name=file.repository_name,
            file_path=file.relative_path,
            source_sha256=file.sha256,
            strategy=chunker.name,
            strategy_version=chunker.version,
            params=chunker.params(),
            start_line=window.start_line,
            end_line=window.end_line,
            fragment_index=window.fragment_index,
        ),
        repository_name=file.repository_name,
        file_path=file.relative_path,
        source_sha256=file.sha256,
        language=file.language,
        chunk_type=chunk_type,
        chunking_strategy=chunker.name,
        chunking_version=chunker.version,
        chunk_index=index,
        start_line=window.start_line,
        end_line=window.end_line,
        content=window.content,
        content_sha256=sha256_text(window.content),
        token_estimate=estimate_tokens(window.content),
        fragment_index=window.fragment_index,
        fragment_count=window.fragment_count,
    )


def chunk_repository(result: IngestionResult, chunker: Chunker) -> ChunkingResult:
    """Chunk every accepted file of an ingestion result and compute statistics.

    Raises ``ChunkingError`` if two chunks share an id (violates the id contract).
    """
    started = time.perf_counter()
    chunks: list[Chunk] = []
    source_lines = 0
    for file in result.files:  # already sorted by relative_path
        chunks.extend(chunker.chunk_file(file))
        source_lines += len(split_lines(file.content))
    elapsed = time.perf_counter() - started

    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ChunkingError("duplicate chunk ids produced; the chunk id contract was violated")

    stats = compute_stats(
        chunks,
        strategy=chunker.name,
        total_files=len(result.files),
        source_lines_total=source_lines,
        elapsed_seconds=elapsed,
    )
    return ChunkingResult(repository_name=result.repository_name, chunks=tuple(chunks), stats=stats)
