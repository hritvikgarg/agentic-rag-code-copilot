"""Baseline chunking strategy "line" (Strategy A): a true structure-blind baseline.

*Every* supported text file - source code, Markdown, JSON, YAML - is cut by the same algorithm:
configurable overlapping line windows, optionally shortened by an estimated token cap (see
``chunking/windows.py``). File language and document structure play no role in chunk
boundaries: Markdown headings are ordinary text. This is the control that the structure-aware
strategy (Milestone 9) is measured against.
"""

from __future__ import annotations

from copilot.chunking.base import build_chunk
from copilot.chunking.windows import split_into_windows, split_lines
from copilot.models.chunk import Chunk, ChunkType
from copilot.models.ingestion import SourceFile

STRATEGY_NAME = "line"
# Bump when the windowing algorithm or the token estimator changes output for the same input and
# parameters. It is part of every chunk id, so stale ids can never be reused across versions.
STRATEGY_VERSION = 1


class LineChunker:
    """Overlapping fixed-size line windows with an estimated-token safety cap."""

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

    @property
    def version(self) -> int:
        return STRATEGY_VERSION

    def params(self) -> dict[str, int | str]:
        return {
            "size_lines": self.size_lines,
            "overlap_lines": self.overlap_lines,
            "max_tokens": self.max_tokens,
        }

    def chunk_file(self, file: SourceFile) -> list[Chunk]:
        windows = split_into_windows(
            split_lines(file.content),
            size=self.size_lines,
            overlap=self.overlap_lines,
            max_tokens=self.max_tokens,
        )
        return [
            build_chunk(file, self, index=i, window=w, chunk_type=ChunkType.LINE_WINDOW)
            for i, w in enumerate(windows)
        ]
