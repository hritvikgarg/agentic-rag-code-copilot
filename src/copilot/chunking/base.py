"""The chunker interface and shared helpers.

Every strategy (baseline line windows now, AST structure-aware in Milestone 9) implements the
same tiny ``Chunker`` protocol, so the indexer, retriever and evaluation harness never need to
know which strategy produced a chunk. That is what makes the two-strategy experiment fair.
"""

from __future__ import annotations

import time
from typing import Protocol

from copilot.chunking.stats import compute_stats
from copilot.chunking.windows import Window
from copilot.models.chunk import Chunk, ChunkingResult, ChunkType, make_chunk_id, sha256_text
from copilot.models.ingestion import IngestionResult, SourceFile
from copilot.utils.tokens import estimate_tokens


class Chunker(Protocol):
    """Turns one ingested file into chunks."""

    @property
    def name(self) -> str:
        """Registry name of the strategy, e.g. ``"line"``. Stored on every chunk."""
        ...

    def params(self) -> dict[str, int | str]:
        """Parameters that affect output; recorded in the index manifest (Milestone 4)."""
        ...

    def chunk_file(self, file: SourceFile) -> list[Chunk]:
        """Chunks of ``file`` in file order (deterministic)."""
        ...


def build_chunk(
    file: SourceFile,
    *,
    strategy: str,
    index: int,
    window: Window,
    chunk_type: ChunkType,
    heading: str | None = None,
) -> Chunk:
    """Create a validated ``Chunk`` from a window of ``file``."""
    content_sha = sha256_text(window.content)
    return Chunk(
        chunk_id=make_chunk_id(
            repository_name=file.repository_name,
            chunking_strategy=strategy,
            file_path=file.relative_path,
            chunk_index=index,
            start_line=window.start_line,
            end_line=window.end_line,
            content_sha256=content_sha,
        ),
        repository_name=file.repository_name,
        file_path=file.relative_path,
        language=file.language,
        chunk_type=chunk_type,
        chunking_strategy=strategy,
        chunk_index=index,
        start_line=window.start_line,
        end_line=window.end_line,
        content=window.content,
        content_sha256=content_sha,
        token_estimate=estimate_tokens(window.content),
        line_fragment=window.fragment,
        heading=heading,
    )


def chunk_repository(result: IngestionResult, chunker: Chunker) -> ChunkingResult:
    """Chunk every accepted file of an ingestion result and compute statistics."""
    started = time.perf_counter()
    chunks: list[Chunk] = []
    for file in result.files:  # already sorted by relative_path
        chunks.extend(chunker.chunk_file(file))
    elapsed = time.perf_counter() - started
    stats = compute_stats(
        chunks,
        strategy=chunker.name,
        total_files=len(result.files),
        elapsed_seconds=elapsed,
    )
    return ChunkingResult(repository_name=result.repository_name, chunks=tuple(chunks), stats=stats)
