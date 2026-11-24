"""Descriptive chunk-size statistics (no retrieval involved).

Used to choose the size-matched arms (Task 4) and to report what each arm's index contains.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from copilot.chunking.stats import percentile
from copilot.chunking.windows import split_lines
from copilot.models.chunk import Chunk
from copilot.models.ingestion import SourceFile


@dataclass(frozen=True)
class SizeProfile:
    """Token and line statistics of a set of chunks, plus redundancy and source coverage."""

    scope: str
    chunk_count: int
    tokens_min: int
    tokens_p25: float
    tokens_median: float
    tokens_mean: float
    tokens_p75: float
    tokens_iqr: float
    tokens_max: int
    lines_median: float
    lines_mean: float
    fragment_chunks: int
    source_lines: int
    represented_lines: int
    coverage: float  # represented / source lines
    non_blank_lines: int
    non_blank_represented: int
    non_blank_coverage: float  # represented non-blank / non-blank source lines
    redundancy: float  # sum of whole-chunk line slots / distinct whole-chunk lines (1.0 = none)

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def profile_chunks(
    chunks: Sequence[Chunk], files: Sequence[SourceFile], *, scope: str
) -> SizeProfile:
    """Profile ``chunks`` against the ``files`` they were cut from (both already scope-filtered)."""
    tokens = sorted(c.token_estimate for c in chunks)
    whole = [c for c in chunks if not c.is_fragment]
    slots = 0
    distinct: set[tuple[str, int]] = set()
    for c in whole:
        slots += c.end_line - c.start_line + 1
        distinct.update((c.file_path, n) for n in range(c.start_line, c.end_line + 1))
    fragment_lines = {(c.file_path, c.start_line) for c in chunks if c.is_fragment}
    represented = len(distinct | fragment_lines)
    source_lines = sum(len(split_lines(f.content)) for f in files)
    by_path = {f.relative_path: split_lines(f.content) for f in files}
    non_blank = sum(1 for lines_ in by_path.values() for line in lines_ if line.strip())
    non_blank_represented = sum(
        1 for path, number in distinct | fragment_lines if by_path[path][number - 1].strip()
    )
    lines = [c.end_line - c.start_line + 1 for c in chunks]
    p25, p75 = percentile(tokens, 25), percentile(tokens, 75)
    return SizeProfile(
        scope=scope,
        chunk_count=len(chunks),
        tokens_min=tokens[0] if tokens else 0,
        tokens_p25=p25,
        tokens_median=float(statistics.median(tokens)) if tokens else 0.0,
        tokens_mean=statistics.fmean(tokens) if tokens else 0.0,
        tokens_p75=p75,
        tokens_iqr=p75 - p25,
        tokens_max=tokens[-1] if tokens else 0,
        lines_median=float(statistics.median(lines)) if lines else 0.0,
        lines_mean=statistics.fmean(lines) if lines else 0.0,
        fragment_chunks=len(chunks) - len(whole),
        source_lines=source_lines,
        represented_lines=represented,
        coverage=represented / source_lines if source_lines else 0.0,
        non_blank_lines=non_blank,
        non_blank_represented=non_blank_represented,
        non_blank_coverage=non_blank_represented / non_blank if non_blank else 0.0,
        redundancy=slots / len(distinct) if distinct else 1.0,
    )


def profile_by_scope(
    chunks: Sequence[Chunk], files: Sequence[SourceFile]
) -> dict[str, SizeProfile]:
    """``{"python": ..., "all": ...}`` profiles (Python-only and repository-wide)."""
    py_files = [f for f in files if f.language == "python"]
    return {
        "python": profile_chunks(
            [c for c in chunks if c.language == "python"], py_files, scope="python"
        ),
        "all": profile_chunks(chunks, files, scope="all"),
    }
