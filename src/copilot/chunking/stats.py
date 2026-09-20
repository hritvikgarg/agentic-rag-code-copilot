"""Chunk statistics: counts, physical-line accounting and token-size distribution."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence

from copilot.models.chunk import Chunk, ChunkStats


def percentile(sorted_values: Sequence[int], q: float) -> float:
    """Nearest-rank percentile of already-sorted values (``q`` in 0..100; 0.0 when empty)."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(q / 100 * len(sorted_values)))
    return float(sorted_values[rank - 1])


def compute_stats(
    chunks: Sequence[Chunk],
    *,
    strategy: str,
    total_files: int,
    source_lines_total: int,
    elapsed_seconds: float,
) -> ChunkStats:
    """Summarise ``chunks`` produced from ``total_files`` files (``source_lines_total`` lines)."""
    tokens = sorted(c.token_estimate for c in chunks)
    files_chunked = len({c.file_path for c in chunks})

    whole = [c for c in chunks if not c.is_fragment]
    fragments = [c for c in chunks if c.is_fragment]

    whole_slots = 0  # sum of line counts of whole-line chunks
    whole_lines: set[tuple[str, int]] = set()
    for c in whole:
        whole_slots += c.end_line - c.start_line + 1
        whole_lines.update((c.file_path, n) for n in range(c.start_line, c.end_line + 1))
    fragmented_lines = {(c.file_path, c.start_line) for c in fragments}

    return ChunkStats(
        strategy=strategy,
        files_chunked=files_chunked,
        files_without_chunks=total_files - files_chunked,
        chunk_count=len(chunks),
        whole_line_chunk_count=len(whole),
        fragment_chunk_count=len(fragments),
        fragmented_line_count=len(fragmented_lines),
        source_lines_total=source_lines_total,
        unique_source_lines_represented=len(whole_lines | fragmented_lines),
        overlap_duplicated_lines=whole_slots - len(whole_lines),
        tokens_mean=statistics.fmean(tokens) if tokens else 0.0,
        tokens_median=float(statistics.median(tokens)) if tokens else 0.0,
        tokens_p95=percentile(tokens, 95),
        tokens_max=tokens[-1] if tokens else 0,
        by_chunk_type=dict(sorted(Counter(c.chunk_type.value for c in chunks).items())),
        by_language=dict(sorted(Counter(c.language for c in chunks).items())),
        elapsed_seconds=elapsed_seconds,
    )
