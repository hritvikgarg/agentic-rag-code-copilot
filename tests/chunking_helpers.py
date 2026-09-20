"""Helpers shared by the chunking tests: build SourceFiles and assert chunk invariants."""

from __future__ import annotations

import hashlib

from copilot.chunking.windows import split_lines
from copilot.models import Chunk, SourceFile


def make_file(
    content: str,
    path: str = "pkg/mod.py",
    language: str = "python",
    extension: str | None = None,
) -> SourceFile:
    raw = content.encode("utf-8")
    return SourceFile(
        repository_name="repo",
        relative_path=path,
        extension=extension or "." + path.rsplit(".", 1)[-1].lower(),
        language=language,
        size_bytes=len(raw),
        content=content,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def assert_chunk_invariants(file: SourceFile, chunks: list[Chunk], *, max_tokens: int) -> None:
    """The invariants every chunking strategy must satisfy for one file."""
    lines = split_lines(file.content)

    # ids unique, indexes dense, order = file order
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    starts = [(c.start_line, c.end_line) for c in chunks]
    assert starts == sorted(starts)

    for c in chunks:
        assert c.file_path == file.relative_path
        assert c.repository_name == file.repository_name
        assert c.language == file.language
        assert 1 <= c.start_line <= c.end_line <= len(lines)
        assert c.source_sha256 == file.sha256
        assert 0 <= c.token_estimate <= max_tokens
        assert c.content.strip() != ""
        if c.is_fragment:
            assert c.start_line == c.end_line  # a fragment keeps its physical line number
            assert 0 <= c.fragment_index < c.fragment_count
            assert c.content in lines[c.start_line - 1]
        else:
            assert c.content == "\n".join(lines[c.start_line - 1 : c.end_line])

    # fragments of one line reassemble to exactly that line
    by_line: dict[int, list] = {}
    for c in chunks:
        if c.is_fragment:
            by_line.setdefault(c.start_line, []).append(c)
    for line_no, pieces in by_line.items():
        assert [c.fragment_index for c in pieces] == list(range(len(pieces)))
        assert {c.fragment_count for c in pieces} == {len(pieces)}
        assert "".join(c.content for c in pieces) == lines[line_no - 1]

    # no whole-line window is contained in another (no near-duplicate cascade)
    windows = [(c.start_line, c.end_line) for c in chunks if not c.is_fragment]
    for i, (a1, b1) in enumerate(windows):
        for j, (a2, b2) in enumerate(windows):
            assert i == j or not (a2 <= a1 and b1 <= b2), (windows[i], windows[j])

    # every non-blank line is covered by some chunk
    covered = {n for c in chunks for n in range(c.start_line, c.end_line + 1)}
    for number, line in enumerate(lines, start=1):
        if line.strip():
            assert number in covered, f"line {number} not covered"
