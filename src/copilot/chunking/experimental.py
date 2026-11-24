"""Experimental chunkers for the Milestone 10 controlled chunking comparison.

These are *controls*, not production strategies. They are deliberately **not** listed in
``available_strategies()``, cannot be selected through ``Settings.chunking_strategy`` and are
never a default; they exist so the comparative experiment can separate "structure" from "chunk
size":

``line_sized`` (Arm C)
    The baseline line-window algorithm (unchanged ``split_into_windows``) with a *different window
    size and overlap for Python files only*. Chosen so that the Python chunk-size distribution is
    close to the structure-aware chunker's. Non-Python files are cut exactly like the baseline.

``ast_merged`` (Arm D)
    The unchanged ``"ast"`` chunker, followed by a deterministic merge of *adjacent, compatible*
    structural chunks of the same file up to ``merge_max_tokens``, so chunks are as large as the
    baseline's while every chunk still starts and ends on an AST-derived boundary. A merged chunk
    holds several units, so it carries **no** symbol metadata (it would be invented otherwise);
    its text is the exact source slice from the first unit's first line to the last unit's last
    line (blank separator lines included).

Both have their own strategy name and version, so their chunk ids can never collide with those of
the production strategies. Non-Python files are chunked by the baseline algorithm with the
production parameters in both, which keeps every difference between the four experimental arms
confined to Python chunking.
"""

from __future__ import annotations

from dataclasses import dataclass

from copilot.chunking import ast_chunker
from copilot.chunking.ast_chunker import AstChunker, AstFileReport
from copilot.chunking.base import SymbolInfo, build_chunk
from copilot.chunking.errors import ChunkingError
from copilot.chunking.windows import Window, split_into_windows, split_lines
from copilot.models.chunk import Chunk, ChunkType
from copilot.models.ingestion import SourceFile
from copilot.utils.tokens import estimate_tokens

SIZED_STRATEGY_NAME = "line_sized"
SIZED_STRATEGY_VERSION = 1
MERGED_STRATEGY_NAME = "ast_merged"
MERGED_STRATEGY_VERSION = 1

EXPERIMENTAL_STRATEGIES = (SIZED_STRATEGY_NAME, MERGED_STRATEGY_NAME)


class SizedLineChunker:
    """Baseline line windows with separate (size, overlap) for Python files (Arm C)."""

    def __init__(
        self,
        *,
        python_size_lines: int,
        python_overlap_lines: int,
        size_lines: int,
        overlap_lines: int,
        max_tokens: int,
    ) -> None:
        for size, overlap in (
            (python_size_lines, python_overlap_lines),
            (size_lines, overlap_lines),
        ):
            if size < 1 or not 0 <= overlap < size:
                raise ValueError("require size >= 1 and 0 <= overlap < size")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.python_size_lines = python_size_lines
        self.python_overlap_lines = python_overlap_lines
        self.size_lines = size_lines
        self.overlap_lines = overlap_lines
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return SIZED_STRATEGY_NAME

    @property
    def version(self) -> int:
        return SIZED_STRATEGY_VERSION

    def params(self) -> dict[str, int | str]:
        return {
            "python_size_lines": self.python_size_lines,
            "python_overlap_lines": self.python_overlap_lines,
            "size_lines": self.size_lines,
            "overlap_lines": self.overlap_lines,
            "max_tokens": self.max_tokens,
        }

    def chunk_file(self, file: SourceFile) -> list[Chunk]:
        is_python = file.language == "python"
        windows = split_into_windows(
            split_lines(file.content),
            size=self.python_size_lines if is_python else self.size_lines,
            overlap=self.python_overlap_lines if is_python else self.overlap_lines,
            max_tokens=self.max_tokens,
        )
        return [
            build_chunk(file, self, index=i, window=w, chunk_type=ChunkType.LINE_WINDOW)
            for i, w in enumerate(windows)
        ]


@dataclass(frozen=True)
class MergeReport:
    """What the merge step did to one file (diagnostics only)."""

    ast_report: AstFileReport
    input_chunks: int
    output_chunks: int
    merged_groups: int  # output chunks that contain more than one original unit


def _window_of(chunk: Chunk) -> Window:
    return Window(
        chunk.start_line,
        chunk.end_line,
        chunk.content,
        fragment_index=chunk.fragment_index,
        fragment_count=chunk.fragment_count,
    )


class MergedAstChunker:
    """The ``"ast"`` chunker with adjacent structural chunks merged up to a token budget (Arm D)."""

    def __init__(
        self,
        *,
        max_tokens: int,
        fallback_size_lines: int,
        fallback_overlap_lines: int,
        merge_max_tokens: int,
    ) -> None:
        if not 1 <= merge_max_tokens <= max_tokens:
            raise ValueError("require 1 <= merge_max_tokens <= max_tokens")
        self.max_tokens = max_tokens
        self.fallback_size_lines = fallback_size_lines
        self.fallback_overlap_lines = fallback_overlap_lines
        self.merge_max_tokens = merge_max_tokens
        self._ast = AstChunker(
            max_tokens=max_tokens,
            fallback_size_lines=fallback_size_lines,
            fallback_overlap_lines=fallback_overlap_lines,
        )

    @property
    def name(self) -> str:
        return MERGED_STRATEGY_NAME

    @property
    def version(self) -> int:
        return MERGED_STRATEGY_VERSION

    def params(self) -> dict[str, int | str]:
        # The wrapped chunker's parameters (incl. its fallback@version) are inputs of the output.
        return {
            **self._ast.params(),
            "merge_max_tokens": self.merge_max_tokens,
            "base": f"{ast_chunker.STRATEGY_NAME}@{ast_chunker.STRATEGY_VERSION}",
        }

    def chunk_file(self, file: SourceFile) -> list[Chunk]:
        return self.analyze(file)[0]

    def analyze(self, file: SourceFile) -> tuple[list[Chunk], MergeReport]:
        """Merged chunks of ``file`` plus a diagnostics report."""
        base, report = self._ast.analyze(file)
        lines = split_lines(file.content)
        if report.segmentation != "ast":  # fallback file: boundaries are the baseline's, unchanged
            out = [self._restamp(file, c, i) for i, c in enumerate(base)]
            return out, MergeReport(report, len(base), len(out), 0)

        groups = self._group(base, lines)
        out: list[Chunk] = []
        merged = 0
        for group in groups:
            if len(group) == 1:
                out.append(self._restamp(file, group[0], len(out)))
                continue
            merged += 1
            start, end = group[0].start_line, group[-1].end_line
            window = Window(start, end, "\n".join(lines[start - 1 : end]))
            out.append(
                build_chunk(
                    file,
                    self,
                    index=len(out),
                    window=window,
                    chunk_type=ChunkType.MERGED,
                    symbol=SymbolInfo(),
                    segmentation="ast",
                )
            )
        self._verify(out, base, lines)
        return out, MergeReport(report, len(base), len(out), merged)

    # -- internals -------------------------------------------------------------------------

    def _restamp(self, file: SourceFile, chunk: Chunk, index: int) -> Chunk:
        """The same unit under this strategy's identity (type, symbols, segmentation preserved)."""
        return build_chunk(
            file,
            self,
            index=index,
            window=_window_of(chunk),
            chunk_type=chunk.chunk_type,
            symbol=SymbolInfo(chunk.symbol_name, chunk.qualified_name, chunk.parent_class),
            segmentation=chunk.segmentation,
        )

    @staticmethod
    def _mergeable(chunk: Chunk) -> bool:
        """Only whole-line chunks bounded by AST statements take part in merging."""
        return chunk.segmentation == "ast" and not chunk.is_fragment

    def _group(self, chunks: list[Chunk], lines: list[str]) -> list[list[Chunk]]:
        """Greedy left-to-right grouping of adjacent, non-overlapping, mergeable chunks."""
        counts = [estimate_tokens(line) for line in lines]
        prefix = [0]
        for c in counts:
            prefix.append(prefix[-1] + c)

        def span_tokens(start: int, end: int) -> int:  # estimated tokens of lines start..end
            return prefix[end] - prefix[start - 1]

        groups: list[list[Chunk]] = []
        for chunk in chunks:
            if groups:
                current = groups[-1]
                last = current[-1]
                if (
                    self._mergeable(chunk)
                    and self._mergeable(current[0])
                    and chunk.start_line > last.end_line
                    and span_tokens(current[0].start_line, chunk.end_line) <= self.merge_max_tokens
                ):
                    current.append(chunk)
                    continue
            groups.append([chunk])
        return groups

    def _verify(self, out: list[Chunk], base: list[Chunk], lines: list[str]) -> None:
        """Self-check: nothing lost or reordered, text is the exact source slice, budget held."""
        previous_end = 0
        for chunk in out:
            if chunk.start_line <= previous_end and chunk.segmentation == "ast":
                raise ChunkingError("merged chunks overlap or are out of order")
            if chunk.chunk_type is ChunkType.MERGED:
                if chunk.content != "\n".join(lines[chunk.start_line - 1 : chunk.end_line]):
                    raise ChunkingError("merged chunk text is not the exact source slice")
                if chunk.token_estimate > self.merge_max_tokens:
                    raise ChunkingError("merged chunk exceeds merge_max_tokens")
            previous_end = chunk.end_line
        if sum(c.end_line - c.start_line + 1 for c in out if not c.is_fragment) < sum(
            c.end_line - c.start_line + 1 for c in base if not c.is_fragment
        ):
            raise ChunkingError("merging lost lines")
