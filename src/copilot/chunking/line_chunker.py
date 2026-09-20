"""Baseline chunking strategy "line" (Strategy A).

Code and data files (JSON/YAML) are cut into overlapping fixed-size line windows; Markdown files
are cut into heading sections. This is intentionally structure-blind for code: it is the baseline
that the structure-aware strategy (Milestone 9) is measured against.
"""

from __future__ import annotations

from copilot.chunking.base import build_chunk
from copilot.chunking.doc_chunker import split_markdown_sections
from copilot.chunking.windows import split_into_windows, split_lines
from copilot.ingestion.languages import DATA_LANGUAGES
from copilot.models.chunk import Chunk, ChunkType
from copilot.models.ingestion import SourceFile

STRATEGY_NAME = "line"
MARKDOWN_LANGUAGE = "markdown"


class LineChunker:
    """Fixed-size overlapping line windows for code/config; heading sections for Markdown."""

    def __init__(self, *, size_lines: int, overlap_lines: int, max_tokens: int) -> None:
        if size_lines < 1 or not 0 <= overlap_lines < size_lines or max_tokens < 1:
            raise ValueError(
                "require size_lines >= 1, 0 <= overlap_lines < size_lines, max_tokens >= 1"
            )
        self.size_lines = size_lines
        self.overlap_lines = overlap_lines
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return STRATEGY_NAME

    def params(self) -> dict[str, int | str]:
        return {
            "size_lines": self.size_lines,
            "overlap_lines": self.overlap_lines,
            "max_tokens": self.max_tokens,
        }

    def chunk_file(self, file: SourceFile) -> list[Chunk]:
        lines = split_lines(file.content)
        if file.language == MARKDOWN_LANGUAGE:
            return self._chunk_markdown(file, lines)
        chunk_type = (
            ChunkType.CONFIG_WINDOW if file.language in DATA_LANGUAGES else ChunkType.CODE_WINDOW
        )
        windows = split_into_windows(
            lines, size=self.size_lines, overlap=self.overlap_lines, max_tokens=self.max_tokens
        )
        return [
            build_chunk(file, strategy=self.name, index=i, window=w, chunk_type=chunk_type)
            for i, w in enumerate(windows)
        ]

    def _chunk_markdown(self, file: SourceFile, lines: list[str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in split_markdown_sections(lines):
            windows = split_into_windows(
                lines[section.start : section.end],
                size=self.size_lines,
                overlap=self.overlap_lines,
                max_tokens=self.max_tokens,
                first_line_number=section.start + 1,
            )
            for window in windows:
                chunks.append(
                    build_chunk(
                        file,
                        strategy=self.name,
                        index=len(chunks),
                        window=window,
                        chunk_type=ChunkType.DOC_SECTION,
                        heading=section.heading,
                    )
                )
        return chunks
