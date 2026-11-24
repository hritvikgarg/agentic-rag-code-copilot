"""Helpers for the structure-aware ("ast") chunking tests."""

from __future__ import annotations

from collections.abc import Sequence

from copilot.chunking import AstChunker, LineChunker
from copilot.chunking.windows import split_lines
from copilot.models import Chunk, SourceFile
from tests.chunking_helpers import assert_chunk_invariants, make_file

DEFAULT_TOKENS = 512


def ast_chunker(max_tokens: int = DEFAULT_TOKENS, size: int = 60, overlap: int = 10) -> AstChunker:
    return AstChunker(
        max_tokens=max_tokens, fallback_size_lines=size, fallback_overlap_lines=overlap
    )


def line_chunker(
    max_tokens: int = DEFAULT_TOKENS, size: int = 60, overlap: int = 10
) -> LineChunker:
    return LineChunker(size_lines=size, overlap_lines=overlap, max_tokens=max_tokens)


def chunk_py(source: str, *, max_tokens: int = DEFAULT_TOKENS, path: str = "pkg/mod.py"):
    """(file, chunks) for ``source`` chunked by the ast strategy; invariants already asserted."""
    file = make_file(source, path)
    chunks = ast_chunker(max_tokens).chunk_file(file)
    assert_chunk_invariants(file, chunks, max_tokens=max_tokens)
    assert_source_reconstructs(file, chunks)
    return file, chunks


def ranges(chunks: Sequence[Chunk]) -> list[tuple[int, int]]:
    return [(c.start_line, c.end_line) for c in chunks]


def by_qualified(chunks: Sequence[Chunk]) -> dict[str | None, list[Chunk]]:
    grouped: dict[str | None, list[Chunk]] = {}
    for c in chunks:
        grouped.setdefault(c.qualified_name, []).append(c)
    return grouped


def assert_source_reconstructs(file: SourceFile, chunks: Sequence[Chunk]) -> None:
    """The chunks, laid back onto their line numbers, rebuild the file.

    Every line is either reproduced exactly by the chunks covering it (fragments of an over-long
    line are re-joined) or is blank in the source. Nothing non-blank is ever missing, and no chunk
    disagrees with the source about what a line says.
    """
    lines = split_lines(file.content)
    seen: dict[int, str] = {}
    fragments: dict[int, dict[int, str]] = {}
    for c in chunks:
        if c.is_fragment:
            assert c.fragment_index is not None
            fragments.setdefault(c.start_line, {})[c.fragment_index] = c.content
            continue
        for offset, text in enumerate(c.content.split("\n")):
            number = c.start_line + offset
            assert seen.get(number, text) == text, f"chunks disagree about line {number}"
            seen[number] = text
    for number, pieces in fragments.items():
        seen[number] = "".join(pieces[i] for i in sorted(pieces))
    for number, line in enumerate(lines, start=1):
        if number in seen:
            assert seen[number] == line, f"line {number} differs from the source"
        else:
            assert not line.strip(), f"non-blank line {number} is missing from every chunk"


def assert_structural_chunks_do_not_overlap(chunks: Sequence[Chunk]) -> None:
    """Chunks cut by AST boundaries (segmentation == "ast") are a tiling: no shared lines."""
    spans = sorted(ranges([c for c in chunks if c.segmentation == "ast"]))
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert end < start, f"structural chunks overlap near line {start}"
